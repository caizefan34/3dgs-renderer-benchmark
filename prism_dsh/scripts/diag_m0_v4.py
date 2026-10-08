"""v4: is the spherical_harmonics backward shape-sensitive? And is the internal
vs standalone sh-gradient gap a uniform scale or structural?"""
import json
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
OUT = "/root/epic05_prism/outputs/diag_m0_v4_bicycle_5000.json"

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

out = {"gpu": GPU}
snap = torch.load(SNAP, map_location="cpu", weights_only=False)
params = {"means": snap["xyz"].cuda().float(), "quats": snap["rotation"].cuda().float(),
          "log_scales": snap["scaling"].cuda().float(),
          "opacity_logits": snap["opacity"].cuda().float().reshape(-1),
          "sh0": snap["shs"][:, :1].cuda().float(), "shN": snap["shs"][:, 1:].cuda().float()}
degree = int(snap["active_sh_degree"])
seq = np.load(SEQP)
cam_ids = list(dict.fromkeys(int(c) for c in seq.tolist()))[:3]
dataset = G.build_dataset("bicycle")
cam_ds, gt = dataset.get_item(cam_ids[0])
cam = G.camera_dict(cam_ds)
if tuple(gt.shape) == (3, cam["height"], cam["width"]):
    gt = gt.permute(1, 2, 0)
gt = gt.float().contiguous()
N = params["means"].shape[0]
sh_all = torch.cat((params["sh0"], params["shN"]), 1).detach()


def rel(a, b):
    a = a.double()
    b = b.double()
    return {"rel_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
            "maxabs": float((a - b).abs().max()), "norm": float(a.norm())}


# ---------- 1: SH op backward shape sensitivity ------------------------------
try:
    with torch.no_grad():
        meta_radii = B.baseline(params, cam, degree)[1]["radii"]
    mask_2d = (meta_radii > 0).all(-1)          # [1,N]
    mask_3d = mask_2d[0][None, None]            # [1,1,N]
    up_c = torch.randn(1, N, 3, device="cuda")
    up_c3 = torch.randn(1, 1, N, 3, device="cuda")

    def sh_grad(means, vm, masks, up):
        s = sh_all.detach().clone().requires_grad_()
        c = (SH(degree, means, vm, s, masks=masks) + .5).clamp_min(0)
        return torch.autograd.grad(c, s, up)[0].detach()

    g_a = sh_grad(params["means"], cam["viewmat"][None], mask_2d, up_c)           # [1] dims
    g_b = sh_grad(params["means"][None, None], cam["viewmat"][None, None], mask_3d, up_c3)  # [1,1]
    g_c = sh_grad(params["means"], cam["viewmat"][None], None, up_c)
    g_d = sh_grad(params["means"][None], cam["viewmat"][None], mask_2d, up_c)     # full 2D coords
    out["1_sh_bwd_shapes"] = {
        "A_plain_vs_B_batchdims": rel(g_a, g_b),
        "A_plain_vs_D_means1dim": rel(g_a, g_d),
        "A_masked_vs_C_nomask": rel(g_a, g_c),
        "rows_masked_out": int((~mask_2d).sum()),
    }
except Exception:
    out["1_sh_bwd_shapes"] = {"error": traceback.format_exc()}

# ---------- 2: internal vs standalone: uniform scale or structural? ----------
try:
    names = ["means", "quats", "log_scales", "opacity_logits", "sh0", "shN"]

    def grads(fn):
        pp = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
        img = fn(pp)
        loss = G.fused_loss(img, gt)
        g = torch.autograd.grad(loss, [pp[k] for k in names], allow_unused=True)
        return img.detach(), loss.detach(), [torch.zeros_like(pp[names[0]]) if x is None else x.detach() for x in g]

    def internal(pp):
        return B.baseline(pp, cam, degree)[0]

    def rgb_route(pp):
        s = torch.cat((pp["sh0"], pp["shN"]), 1)
        cc = (SH(degree, pp["means"], cam["viewmat"][None], s, masks=mask_2d) + .5).clamp_min(0)
        _r, _a, _m = rasterization(
            means=pp["means"], quats=F.normalize(pp["quats"], dim=-1), scales=pp["log_scales"].exp(),
            opacities=pp["opacity_logits"].sigmoid(), colors=cc,
            viewmats=cam["viewmat"][None], Ks=cam["K"][None],
            width=cam["width"], height=cam["height"], sh_degree=None,
            eps2d=.3, radius_clip=0., tile_size=16, packed=False, absgrad=True,
            render_mode="RGB", rasterize_mode="classic")
        return _r[0].clamp(0, 1)

    img_i, loss_i, g_i = grads(internal)
    img_r, loss_r, g_r = grads(rgb_route)
    res = {"loss_internal": float(loss_i), "loss_rgb": float(loss_r),
           "img_maxabs": float((img_i - img_r).abs().max()), "groups": {}}
    for k, a, b in zip(names, g_i, g_r):
        res["groups"][k] = rel(a, b)
        if k in ("sh0", "shN"):
            aa = a.double().flatten()
            bb = b.double().flatten()
            thresh = bb.abs().quantile(0.999)
            sel = bb.abs() > thresh
            ratio = (aa[sel] / bb[sel])
            res["groups"][k]["ratio_on_top0.1pct"] = {
                "n": int(sel.sum()), "median": float(ratio.median()),
                "mean": float(ratio.mean()), "std": float(ratio.std()),
                "q05": float(ratio.quantile(0.05)), "q95": float(ratio.quantile(0.95)),
                "cos_angle": float((aa @ bb) / (aa.norm() * bb.norm()).clamp_min(1e-30))}
            idx = aa.abs().topk(5).indices
            res["groups"][k]["top5_coords"] = [
                {"flat": int(i), "internal": float(aa[i]), "rgb": float(bb[i]),
                 "ratio": float(aa[i] / bb[i]) if bb[i] != 0 else None} for i in idx.tolist()]
    out["2_ratio_analysis"] = res
except Exception:
    out["2_ratio_analysis"] = {"error": traceback.format_exc()}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str)[:9000])
