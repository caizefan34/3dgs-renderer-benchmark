import importlib.util
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
spec = importlib.util.spec_from_file_location(
    "p17", "/root/epic05_p17/p17_p0_oracle.py")
p17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p17)

import numpy as np
import torch

t0 = time.time()
ctx = p17.SceneCtx("room")
ci = 12
ctx.cur_ci = ci
cam, gt = ctx.cam_gt(ci)
H0, W0 = int(cam.image_height), int(cam.image_width)
K0 = cam.K.detach().cpu().numpy().astype(np.float64)
K0t = torch.as_tensor(K0, device="cuda", dtype=torch.float32)

vis1 = ctx.visible_ids(cam, K0, W0, H0, p17.EPS2D_FULL)
rad1, _, _, conics1 = ctx.producer_full(cam, K0, W0, H0, p17.EPS2D_FULL)
vis_rows = rad1[:, 0] > 0
sig_min1, _ = p17.sigma_min_from_conics(conics1)
sig_min1[~vis_rows] = 0.0
d_full = p17.assign_d(sig_min1.cpu(), 2.0)
print("assignment:", {f"d{d}": int((d_full == d).sum().item())
                      for d in (1, 2, 4, 8)})

coarse, coarse_vis, coarse_counts = {}, {}, {}
N = ctx.N
gt_np = gt.cpu().numpy()
for d in p17.DS:
    Hd, Wd = H0 // d, W0 // d
    eps_d = p17.EPS2D_FULL / (d * d)
    gt_c = torch.as_tensor(p17.area_avg_2d(gt_np, d), device="cuda",
                           dtype=torch.float32)
    _, g_d, fr, loss, meta = p17.render_backward_grads(
        ctx, cam, torch.as_tensor(p17.scale_K(K0, d), device="cuda",
                                  dtype=torch.float32), Wd, Hd, eps_d,
        gt_coarse=gt_c)
    vis_d = ctx.visible_ids(cam, p17.scale_K(K0, d), Wd, Hd, eps_d)
    c_d, _ = ctx.intersect_counts(cam, p17.scale_K(K0, d), Wd, Hd, eps_d, vis_d)
    coarse[d] = {"grads": {k: v.reshape(N, -1) for k, v in g_d.items()},
                 "loss": loss}
    coarse_vis[d] = vis_d
    coarse_counts[d] = c_d
    print(f"d={d}: n_vis={vis_d.shape[0]} loss={loss:.6f}")

_, ref_flat, fr1, loss1, meta1 = p17.render_backward_grads(
    ctx, cam, K0t, W0, H0, p17.EPS2D_FULL)
ref_pg = ctx.grad_groups()
hyb = {k: torch.zeros_like(ref_pg[k]) for k in ref_pg}
assigned = torch.zeros(N, dtype=torch.int64)
cost_by_d = torch.zeros(N, dtype=torch.float64)
vis1_t = torch.as_tensor(vis1, dtype=torch.int64)
cnt1_full = torch.zeros(N, dtype=torch.float64)
cnt1_full[vis1_t] = torch.as_tensor(
    ctx.intersect_counts(cam, K0, W0, H0, p17.EPS2D_FULL, vis1)[0],
    dtype=torch.float64)
for d in p17.DS:
    sel = (d_full == d)
    if not bool(sel.any()):
        continue
    vis_d = torch.as_tensor(coarse_vis[d], dtype=torch.int64)
    cnt_d_full = torch.zeros(N, dtype=torch.float64)
    cnt_d_full[vis_d] = torch.as_tensor(coarse_counts[d], dtype=torch.float64)
    for k in ref_pg:
        if k in ("sh0", "shN"):
            continue
        g_d_full = torch.zeros_like(ref_pg[k])
        g_d_full[vis_d] = coarse[d]["grads"][k][: vis_d.numel()]
        hyb[k] = torch.where(sel.unsqueeze(1), g_d_full, hyb[k])
    assigned[sel] = d
    cost_by_d[sel] = cnt_d_full[sel]
C_oracle = float(cost_by_d.sum()) / float(cnt1_full.sum())
print("C_oracle:", C_oracle)

# ---- population-level cosine per (group, d) ----
print("\n--- per-(group, d) population cosine (ref full-res vs coarse d) ---")
for d in (2, 4, 8):
    sel = (d_full == d)
    vis_d = torch.as_tensor(coarse_vis[d], dtype=torch.int64)
    in_vis = torch.zeros(N, dtype=torch.bool)
    in_vis[vis_d] = True
    for k in ("means", "scales", "quats", "opacity", "shs"):
        a = ref_pg[k][sel].reshape(-1)
        b = coarse[d]["grads"][k][sel].reshape(-1)
        # visible subset
        sv = sel & in_vis
        if int(sv.sum()) > 0:
            av = ref_pg[k][sv].reshape(-1)
            bv = coarse[d]["grads"][k][sv].reshape(-1)
            print(f"d={d} {k:8s} pop_n={int(sel.sum()):6d} "
                  f"vis_n={int(sv.sum()):6d} "
                  f"cos_pop={p17.cosine(a, b):.4f} "
                  f"cos_vis={p17.cosine(av, bv):.4f}")
        else:
            print(f"d={d} {k:8s} pop_n={int(sel.sum()):6d} vis_n=0")

# ---- step (exactly as in the oracle main loop) ----
hyb_step = {k: v.reshape(-1) for k, v in hyb.items()}
L_before_d = p17.dense_adam_step(ctx, cam, K0, W0, H0)
L_after_dense = p17.poststep_eval(ctx, cam, K0, W0, H0, gt)
L_before_h = p17.apply_hybrid_adam_step(ctx, cam, K0, W0, H0, hyb_step)
L_after_hyb = p17.poststep_eval(ctx, cam, K0, W0, H0, gt)
assert abs(L_before_d - L_before_h) < 1e-12, "L_before mismatch"
G_dense = L_before_d - L_after_dense
G_hyb = L_before_h - L_after_hyb
ref_improving = bool(G_dense > 0.0)
R_step = (G_hyb / max(G_dense, 1e-12)) if ref_improving else None
print(f"\ndense: Lb={L_before_d:.8f} La={L_after_dense:.8f} "
      f"G={G_dense:.3e}")
print(f"hyb:   Lb={L_before_h:.8f} La={L_after_hyb:.8f} "
      f"G={G_hyb:.3e}")
print(f"ref_improving={ref_improving} R_step={R_step}")

# ---- audit (per prereg: mean over 4 audit cameras of (L_hyb - L_dense)/L0)
audit_ids = [50, 130, 210, 290]
audit_ids = [aid for aid in audit_ids if aid < ctx.n_cameras]
ctx.restore_initial()
L_audit_before = p17.audit_losses(ctx, audit_ids)
# dense step -> audit
p17.dense_adam_step(ctx, cam, K0, W0, H0)
L_audit_after_dense = p17.audit_losses(ctx, audit_ids)
# hybrid step -> audit (from restored initial)
ctx.restore_initial()
p17.apply_hybrid_adam_step(ctx, cam, K0, W0, H0, hyb_step)
L_audit_after_hyb = p17.audit_losses(ctx, audit_ids)
P_audit_vals = [
    (L_audit_after_hyb[i] - L_audit_after_dense[i])
    / (L_audit_before[i] + 1e-30) for i in range(len(audit_ids))]
import numpy as _np
print(f"P_audit per cam: {[f'{v:.3e}' for v in P_audit_vals]}")
print(f"P_audit mean: {_np.mean(P_audit_vals):.3e}")
print("DONE", f"({time.time()-t0:.1f}s)")
