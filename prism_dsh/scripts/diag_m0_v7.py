"""v7: (A) torch activation boundary semantics; (B) sh-grad activation variants vs
internal route (+ rasterizer v_colors route comparison); (C) covariance arithmetic
mirror candidates vs the kernel's own quat_scale_to_covar_preci binding.

Predicts the M0 rerun outcome for candidate patches. Read-only; no bundle edits."""
import json
import math
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

BUNDLE = "/root/epic05_prism/prism_gs"
GX = "/root/epic05_grc3d/scripts"
SNAP = "/root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt"
SEQP = "/root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy"
OUT = "/root/epic05_prism/outputs/diag_m0_v7_bicycle_5000.json"

GPU = int(os.environ.get("PRISM_GPU", "2"))
sys.path.insert(0, GX)
sys.path.insert(0, BUNDLE)
import gx_common as G  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
trainer, ident = G.boot_gmain(GPU)
from prism import gsplat_backend as B  # noqa: E402
from prism.geometry import covariance  # noqa: E402
from gsplat import rasterization  # noqa: E402
from gsplat.cuda._wrapper import spherical_harmonics as SH  # noqa: E402
from gsplat.cuda._wrapper import quat_scale_to_covar_preci as KS2C  # noqa: E402

out = {"gpu": GPU}


def save():
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)


# ---------- A: torch activation boundary semantics ---------------------------
try:
    base = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 2.0, -3.0], requires_grad=True)
    res = {}
    for name, fn in [("clamp_min", lambda t: t.clamp_min(0)),
                     ("relu", lambda t: t.relu()),
                     ("where", lambda t: torch.where(t > 0, t, torch.zeros_like(t)))]:
        x = base.detach().clone().requires_grad_()
        fn(x).sum().backward()
        res[name] = {"grad": [float(v) for v in x.grad.tolist()]}
    out["A_activation_boundary"] = res
except Exception:
    out["A_activation_boundary"] = {"error": traceback.format_exc()}
save()

# ---------- setup ------------------------------------------------------------
snap = torch.load(SNAP, map_location="cpu", weights_only=False)
params = {"means": snap["xyz"].cuda().float(), "quats": snap["rotation"].cuda().float(),
          "log_scales": snap["scaling"].cuda().float(),
          "opacity_logits": snap["opacity"].cuda().float().reshape(-1),
          "sh0": snap["shs"][:, :1].cuda().float(), "shN": snap["shs"][:, 1:].cuda().float()}
degree = int(snap["active_sh_degree"])
seq = np.load(SEQP)
cam_ids = list(dict.fromkeys(int(c) for c in seq.tolist()))[:1]
dataset = G.build_dataset("bicycle")
cam_ds, gt = dataset.get_item(cam_ids[0])
cam = G.camera_dict(cam_ds)
if tuple(gt.shape) == (3, cam["height"], cam["width"]):
    gt = gt.permute(1, 2, 0)
gt = gt.float().contiguous()
N = params["means"].shape[0]
sh_all = torch.cat((params["sh0"], params["shN"]), 1).detach()
K = sh_all.shape[1]

with torch.no_grad():
    img_q, meta_q = B.baseline(params, cam, degree)
mask_any = (meta_q["radii"] > 0).any(-1)   # same predicate as B.state
KNOWN_ROWS = [402712, 409556, 66212, 781122, 785049, 1010530]


def rel(a, b):
    a = a.double(); b = b.double()
    return {"rel_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
            "maxabs": float((a - b).abs().max()), "norm": float(a.norm())}


# ---------- B1: pre-relu boundary census -------------------------------------
try:
    with torch.no_grad():
        pre = SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_any)
        if pre.dim() == 2:
            pre = pre[None]
        col_pre = pre + .5
        vis = mask_any[0]
        eq0 = (col_pre[0] == 0.0)
        lt0 = (col_pre[0] < 0.0)
        gt0 = (col_pre[0] > 0.0)
        res = {
            "shape": list(pre.shape),
            "visible_rows": int(vis.sum()),
            "entries_eq_zero_visible": int((eq0 & vis[:, None]).sum()),
            "entries_lt_zero_visible": int((lt0 & vis[:, None]).sum()),
            "entries_gt_zero_visible": int((gt0 & vis[:, None]).sum()),
            "channel_groups_eq_zero_visible": int((eq0.any(-1) & vis).sum()),
            "channel_groups_lt_zero_visible": int((lt0.any(-1) & vis).sum()),
            "known_rows": [],
        }
        for r in KNOWN_ROWS:
            res["known_rows"].append({
                "row": r, "visible": bool(vis[r]),
                "pre": [float(v) for v in pre[0, r].tolist()],
                "col_pre": [float(v) for v in col_pre[0, r].tolist()],
                "eq0": [bool(v) for v in eq0[r].tolist()],
            })
    out["B1_pre_census"] = res
except Exception:
    out["B1_pre_census"] = {"error": traceback.format_exc()}
save()

# ---------- B2: rasterizer v_colors: high-level vs ops-level -----------------
try:
    names = ["means", "quats", "log_scales", "opacity_logits", "sh0", "shN"]

    def grads(fn):
        pp = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
        img = fn(pp)
        loss = G.fused_loss(img, gt)
        g = torch.autograd.grad(loss, [pp[k] for k in names])
        return [x.detach() for x in g]

    def internal(pp):
        return B.baseline(pp, cam, degree)[0]

    def rgb_route(activation):
        def f(pp):
            s = torch.cat((pp["sh0"], pp["shN"]), 1)
            cc = activation(SH(degree, pp["means"], cam["viewmat"][None], s, masks=mask_any) + .5)
            _r, _a, _m = rasterization(
                means=pp["means"], quats=F.normalize(pp["quats"], dim=-1), scales=pp["log_scales"].exp(),
                opacities=pp["opacity_logits"].sigmoid(), colors=cc,
                viewmats=cam["viewmat"][None], Ks=cam["K"][None],
                width=cam["width"], height=cam["height"], sh_degree=None,
                eps2d=.3, radius_clip=0., tile_size=16, packed=False, absgrad=True,
                render_mode="RGB", rasterize_mode="classic")
            return _r[0].clamp(0, 1)
        return f

    print("internal grads", flush=True)
    g_i = grads(internal)
    gi_all = torch.cat((g_i[4], g_i[5]), 1)
    res = {}
    for name, act in [("clamp", lambda t: t.clamp_min(0)),
                      ("relu", lambda t: t.relu()),
                      ("where", lambda t: torch.where(t > 0, t, torch.zeros_like(t)))]:
        print("variant", name, flush=True)
        g_v = grads(rgb_route(act))
        gv_all = torch.cat((g_v[4], g_v[5]), 1)
        entry = {}
        for gi, gv, nm in zip(g_i, g_v, names):
            entry[nm] = rel(gi, gv)
        d = (gi_all != gv_all)
        entry["zero_entry_mismatch"] = int((d & (gi_all == 0)).sum())
        entry["m0_style_sh_pass"] = bool(entry["sh0"]["rel_l2"] <= 1e-3 and entry["shN"]["rel_l2"] <= 1e-3)
        res[name] = entry
        del g_v, gv_all, d
        save()
    out["B3_activation_variants"] = res

    # v_colors route comparison (colors as leaf through both rasterizer entries)
    with torch.no_grad():
        cc0 = (SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_any) + .5).clamp_min(0)
    c_hi = cc0.detach().clone().requires_grad_()
    img_hi, _a, _m = rasterization(
        means=params["means"], quats=F.normalize(params["quats"], dim=-1), scales=params["log_scales"].exp(),
        opacities=params["opacity_logits"].sigmoid(), colors=c_hi,
        viewmats=cam["viewmat"][None], Ks=cam["K"][None],
        width=cam["width"], height=cam["height"], sh_degree=None,
        eps2d=.3, radius_clip=0., tile_size=16, packed=False, absgrad=True,
        render_mode="RGB", rasterize_mode="classic")
    loss_hi = G.fused_loss(img_hi[0].clamp(0, 1), gt)
    vc_hi = torch.autograd.grad(loss_hi, c_hi)[0].detach()
    del c_hi, img_hi

    st = B.state(params, cam, degree)
    st_d = B.detach_state(st)
    c_lo = st_d["colors"].detach().clone().requires_grad_()
    img_lo = B.draw(dict(st_d, colors=c_lo))
    loss_lo = G.fused_loss(img_lo, gt)
    vc_lo = torch.autograd.grad(loss_lo, c_lo)[0].detach()
    out["B4_vcolors_routes"] = {
        "img_maxabs": float((img_lo - img_q).abs().max()),
        "vcolors": rel(vc_hi, vc_lo),
        "nonzero_hi": int((vc_hi != 0).sum()), "nonzero_lo": int((vc_lo != 0).sum()),
        "zero_mismatch": int(((vc_hi == 0) != (vc_lo == 0)).sum()),
    }
    del g_i, gi_all, vc_hi, vc_lo
    save()
except Exception:
    out["B3_activation_variants"] = {"error": traceback.format_exc()}
    save()

# ---------- C: covariance arithmetic mirror candidates -----------------------
try:
    scales = params["log_scales"].exp()
    K0, _ = KS2C(params["quats"].contiguous(), scales.contiguous(), True, False, False)
    V0 = covariance(params["quats"], params["log_scales"])


    def fma(a, b, c):
        return (a.double() * b.double() + c.double()).float()


    def mirror(quats, sc, mode="plain"):
        q = quats
        x = q[..., 1]; y = q[..., 2]; z = q[..., 3]; w = q[..., 0]
        if mode == "fma_len":
            len2 = fma(x, x, fma(y, y, fma(z, z, w * w)))
        else:
            len2 = x * x + y * y + z * z + w * w
        inv = torch.rsqrt(len2)
        x = x * inv; y = y * inv; z = z * inv; w = w * inv
        x2 = x * x; y2 = y * y; z2 = z * z
        xy = x * y; xz = x * z; yz = y * z
        wx = w * x; wy = w * y; wz = w * z
        if mode == "fma_R":
            r00 = fma(torch.full_like(z2, -2.0), y2 + z2, torch.ones_like(z2))
            r11 = fma(torch.full_like(z2, -2.0), x2 + z2, torch.ones_like(z2))
            r22 = fma(torch.full_like(z2, -2.0), x2 + y2, torch.ones_like(z2))
        else:
            r00 = 1 - 2 * (y2 + z2); r11 = 1 - 2 * (x2 + z2); r22 = 1 - 2 * (x2 + y2)
        r01 = 2 * (xy - wz); r02 = 2 * (xz + wy)
        r10 = 2 * (xy + wz); r12 = 2 * (yz - wx)
        r20 = 2 * (xz - wy); r21 = 2 * (yz + wx)
        R = torch.stack((r00, r01, r02, r10, r11, r12, r20, r21, r22), -1).reshape(q.shape[:-1] + (3, 3))
        M = R * sc[..., None, :]

        def c3(i, j):
            a = M[..., i, 0] * M[..., j, 0]
            b = M[..., i, 1] * M[..., j, 1]
            c = M[..., i, 2] * M[..., j, 2]
            if mode == "fma_cov":
                return fma(M[..., i, 2], M[..., j, 2], fma(M[..., i, 1], M[..., j, 1], a))
            if mode == "fma_cov2":
                return fma(M[..., i, 1], M[..., j, 1], a) + c
            return (a + b) + c

        cov = torch.stack((c3(0, 0), c3(0, 1), c3(0, 2), c3(1, 0), c3(1, 1), c3(1, 2),
                           c3(2, 0), c3(2, 1), c3(2, 2)), -1).reshape(q.shape[:-1] + (3, 3))
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        return cov

    def diff_stats(a, b):
        d = (a - b).abs()
        r = d / b.abs().clamp_min(1e-30)
        return {"bitwise_equal": int((a == b).all(-1).all(-1).sum()),
                "rows": int(a.shape[0]),
                "maxabs": float(d.max()), "maxrel": float(r.max()),
                "rel_gt1e-6": int((r.amax(dim=(-1, -2)) > 1e-6).sum()),
                "rel_gt1e-4": int((r.amax(dim=(-1, -2)) > 1e-4).sum())}

    cand_list = [("V0_current", V0)]
    for mode in ("plain", "fma_len", "fma_R", "fma_cov", "fma_cov2"):
        cand_list.append((mode, mirror(params["quats"], scales, mode)))
    res = {}
    for name, V in cand_list:
        res[name] = diff_stats(K0, V)
        with torch.no_grad():
            img_v = B.baseline(params, cam, degree, covars=V)[0]
        res[name]["render_vs_original_maxabs"] = float((img_v - img_q).abs().max())
        del img_v
        save()
    with torch.no_grad():
        img_k0 = B.baseline(params, cam, degree, covars=K0)[0]
    res["K0_route_render_vs_original_maxabs"] = float((img_k0 - img_q).abs().max())
    out["C_covar"] = res
except Exception:
    out["C_covar"] = {"error": traceback.format_exc()}
save()
print("DONE", flush=True)
