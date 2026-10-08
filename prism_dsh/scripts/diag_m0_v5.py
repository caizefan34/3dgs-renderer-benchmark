"""v5: SH-bwd shape sensitivity (correct dims), internal-vs-op gradient ratio
structure, conic diff distribution, radius-flip causality via patched rebuild."""
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
OUT = "/root/epic05_prism/outputs/diag_m0_v5_bicycle_5000.json"

GPU = int(os.environ.get("PRISM_GPU", "2"))
sys.path.insert(0, GX)
sys.path.insert(0, BUNDLE)
import gx_common as G  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
trainer, ident = G.boot_gmain(GPU)
from prism import gsplat_backend as B  # noqa: E402
from prism.geometry import covariance, conic_to_cov, cov_to_conic, screen_pack, screen_unpack  # noqa: E402
from gsplat import rasterization  # noqa: E402
from gsplat.cuda._wrapper import spherical_harmonics as SH  # noqa: E402
from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode, rasterize_to_pixels  # noqa: E402

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


def rebuild(meta, colors, radii=None, tile=16):
    uv = meta["means2d"]
    co = meta["conics"]
    depth = meta["depths"]
    opac = meta["opacities"]
    if opac.ndim == 1:
        opac = opac[None]
    rr = meta["radii"] if radii is None else radii
    tw = math.ceil(cam["width"] / tile)
    th = math.ceil(cam["height"] / tile)
    _, isect, flat = isect_tiles(uv, rr, depth, tile, tw, th, packed=False)
    off = isect_offset_encode(isect, 1, tw, th)
    img, _ = rasterize_to_pixels(uv, co, colors, opac, cam["width"], cam["height"],
                                 tile, off, flat, packed=False, absgrad=False)
    return img[0].clamp(0, 1)


with torch.no_grad():
    img_q, meta_q = B.baseline(params, cam, degree)
    img_c, meta_c = B.baseline(params, cam, degree, covars=covariance(params["quats"], params["log_scales"]))
mask_all = (meta_q["radii"] > 0).all(-1)
colors = (SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_all) + .5).clamp_min(0)

# ---------- 1: SH bwd shape variants (viewmats must be one dim above means) --
try:
    up_small = torch.randn(1, N, 3, device="cuda")
    up_1 = torch.randn(1, 1, N, 3, device="cuda")

    def sh_grad(means, vm, masks, up):
        s = sh_all.detach().clone().requires_grad_()
        c = (SH(degree, means, vm, s, masks=masks) + .5).clamp_min(0)
        return torch.autograd.grad(c, s, up)[0].detach()

    g_A = sh_grad(params["means"], cam["viewmat"][None], mask_all, up_small)                  # [N,3] + [1,4,4]
    m1 = mask_all[0][None]                                                                   # [1,N]
    g_B = sh_grad(params["means"][None], cam["viewmat"][None, None], m1, up_1)               # [1,N,3] + [1,1,4,4]
    m2 = mask_all[0][None, None]
    g_C = sh_grad(params["means"][None, None], cam["viewmat"][None, None, None], m2,
                  up_1[None])                                                                # [1,1,N,3] + [1,1,1,4,4]
    out["1_sh_bwd_shapes"] = {"A_vs_B_batch1": rel(g_A, g_B), "A_vs_C_batch2": rel(g_A, g_C),
                              "B_vs_C_batch2": rel(g_B, g_C), "rows_masked_out": int((~mask_all).sum())}
except Exception:
    out["1_sh_bwd_shapes"] = {"error": traceback.format_exc()}

# ---------- 2: internal vs standalone grad ratio structure -------------------
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
        cc = (SH(degree, pp["means"], cam["viewmat"][None], s, masks=mask_all) + .5).clamp_min(0)
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
            kk = min(2000, bb.numel())
            top = bb.abs().topk(kk).indices
            ratio = aa[top] / bb[top]
            res["groups"][k]["ratio_top2000"] = {
                "median": float(ratio.median()), "mean": float(ratio.mean()),
                "std": float(ratio.std()), "q05": float(ratio.quantile(0.05)),
                "q95": float(ratio.quantile(0.95)),
                "cos_angle_all": float((aa @ bb) / (aa.norm() * bb.norm()).clamp_min(1e-30)),
                "n_nonzero_rgb": int((bb != 0).sum())}
            res["groups"][k]["top5_coords"] = [
                {"flat": int(i), "internal": float(aa[i]), "rgb": float(bb[i]),
                 "ratio": float(aa[i] / bb[i]) if bb[i] != 0 else None} for i in top[:5].tolist()]
    out["2_ratio"] = res
except Exception:
    out["2_ratio"] = {"error": traceback.format_exc()}

# ---------- 3: cov route conic diff distribution -----------------------------
try:
    rq, rc = meta_q["radii"][0], meta_c["radii"][0]
    valid = (rq > 0).any(-1) & (rc > 0).any(-1)
    cq, cc = meta_q["conics"][0][valid], meta_c["conics"][0][valid]
    drel = ((cq - cc).abs() / cq.abs().clamp_min(1e-20)).amax(-1)
    dab = (cq - cc).abs().amax(-1)
    top = drel.topk(6).indices
    scales = params["log_scales"].exp()
    out["3_conics"] = {
        "valid_rows": int(valid.sum()),
        "rel_median": float(drel.median()), "rel_q99": float(drel.quantile(0.99)),
        "rel_max": float(drel.max()),
        "abs_median": float(dab.median()), "abs_max": float(dab.max()),
        "rows_rel_gt_1e-3": int((drel > 1e-3).sum()),
        "rows_rel_gt_1e-5": int((drel > 1e-5).sum()),
        "worst": [{"row_valid_idx": int(i),
                   "rel": float(drel[i]), "abs_conic": float(dab[i]),
                   "conic_quat": cq[i].tolist(), "conic_cov": cc[i].tolist(),
                   "scale": scales[valid.nonzero().flatten()[i]].tolist(),
                   "opacity": float(meta_q["opacities"][0][valid][i]),
                   "radii_quat": rq[valid][i].tolist(), "radii_cov": rc[valid][i].tolist()}
                  for i in top.tolist()],
    }
except Exception:
    out["3_conics"] = {"error": traceback.format_exc()}

# ---------- 4: radius-flip causality -----------------------------------------
try:
    flip = (rq != rc).any(-1)
    img_q_rebuild = rebuild(meta_q, colors)
    img_c_rebuild = rebuild(meta_c, colors)
    rad_patch = rc.clone()
    rad_patch[flip] = rq[flip]
    img_c_patch = rebuild(meta_c, colors, radii=rad_patch[None])
    out["4_causality"] = {
        "flip_rows": int(flip.sum()),
        "sanity_rebuildQ_vs_imgQ": float((img_q_rebuild - img_q).abs().max()),
        "sanity_rebuildC_vs_imgC": float((img_c_rebuild - img_c).abs().max()),
        "rebuildC_patched_vs_imgQ": float((img_c_patch - img_q).abs().max()),
        "rebuildC_unpatched_vs_imgQ": float((img_c_rebuild - img_q).abs().max()),
    }
except Exception:
    out["4_causality"] = {"error": traceback.format_exc()}

# ---------- 5: zero-override radius flips & causality ------------------------
try:
    image, st = B.state(params, cam, degree)
    meta = st["meta"]
    radii = meta["radii"][0]
    rows_ok = (radii > 0).any(-1)
    cov2d = torch.zeros(N, 2, 2, device="cuda")
    cov2d[rows_ok] = conic_to_cov(meta["conics"][0][rows_ok])
    # adapter draw() recomputation path, but full-N (not just 256)
    zfull = screen_pack(meta["means2d"][0], cov2d)
    uv_o, cov_o = screen_unpack(zfull)
    opac = meta["opacities"]
    if opac.ndim == 2:
        opac = opac[0]
    extend = (2 * torch.log((opac * 255).clamp_min(1))).sqrt().clamp_max(3.33)
    rr = torch.ceil(extend[:, None] * cov_o.diagonal(dim1=-2, dim2=-1).sqrt()).to(radii.dtype)
    rr = torch.where(rows_ok[:, None], rr, torch.zeros_like(rr))
    flip2 = (rr != radii).any(-1)
    rr_patch = rr.clone()
    rr_patch[flip2] = radii[flip2]
    img_orig = rebuild(meta, st["colors"])
    img_repatch = rebuild(meta, st["colors"], radii=rr_patch[None])
    img_rr = rebuild(meta, st["colors"], radii=rr[None])
    # float64 tie-break distance for flipped rows
    fl = flip2.nonzero().flatten()
    thr = []
    for i in fl[:10].tolist():
        p_a = (extend[i].double() * cov2d[i].diagonal().double().sqrt())
        p_b = (extend[i].double() * cov_o[i].diagonal().double().sqrt())
        thr.append({"row": int(i), "dims": [float(x) for x in p_a.tolist()],
                    "dims_rt": [float(x) for x in p_b.tolist()],
                    "dist_to_int": [float(abs(x - round(x))) for x in p_b.tolist()]})
    out["5_zero_override"] = {
        "flip_rows_allN": int(flip2.sum()),
        "flip_values": {str(int(v)): int(c) for v, c in
                        zip(*np.unique((rr[flip2] - radii[flip2]).cpu().numpy(), return_counts=True))},
        "img_orig_vs_rebuild": float((img_orig - image).abs().max()),
        "img_orig_vs_rr": float((img_orig - img_rr).abs().max()),
        "img_orig_vs_rr_patched": float((img_orig - img_repatch).abs().max()),
        "flip_tie_distances": thr,
    }
except Exception:
    out["5_zero_override"] = {"error": traceback.format_exc()}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str)[:9000])
