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

# coarse probes d=2/4/8
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

# full-res reference backward
_, ref_flat, fr1, loss1, meta1 = p17.render_backward_grads(
    ctx, cam, K0t, W0, H0, p17.EPS2D_FULL)
ref_pg = ctx.grad_groups()

# hybrid assembly
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
pop = torch.zeros(N, dtype=torch.bool)
pop[torch.where(assigned > 1)[0]] = True
mtr = p17.metrics_for(ctx, pop, ref_pg, hyb)
print("metrics:", {k: (f"{v['cosine']:.5f}" if v['cosine'] is not None else "NA")
                   for k, v in mtr.items()})

# steps
hyb_step = {k: v.reshape(-1) for k, v in hyb.items()}
Lb = p17.dense_adam_step(ctx, cam, K0, W0, H0)
La = p17.poststep_eval(ctx, cam, K0, W0, H0, gt)
print(f"dense: Lb={Lb:.8f} La={La:.8f} G={Lb-La:.3e}")
Lbh = p17.apply_hybrid_adam_step(ctx, cam, K0, W0, H0, hyb_step)
Lah = p17.poststep_eval(ctx, cam, K0, W0, H0, gt)
print(f"hyb:   Lb={Lbh:.8f} La={Lah:.8f} G={Lbh-Lah:.3e}")
print("DONE", f"({time.time()-t0:.1f}s)")
