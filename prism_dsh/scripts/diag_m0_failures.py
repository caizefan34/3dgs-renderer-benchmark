"""Diagnose the three BLOCKED_M0 sub-checks: covars re-render, zero-override
rebuilt bounds, sh0/shN gradient mismatch. Read-only diagnosis, no fixes."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BUNDLE = "/root/epic05_prism/prism_gs"
GX = "/root/epic05_grc3d/scripts"
SNAP = "/root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt"
SEQP = "/root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy"
OUT = "/root/epic05_prism/outputs/diag_m0_bicycle_5000.json"

GPU = int(os.environ.get("PRISM_GPU", "2"))
sys.path.insert(0, GX)
sys.path.insert(0, BUNDLE)
import gx_common as G  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
trainer, ident = G.boot_gmain(GPU)
from prism import gsplat_backend as B  # noqa: E402
from prism.geometry import covariance, conic_to_cov, cov_to_conic, project, unpack_geometry, pack_geometry  # noqa: E402
from gsplat import rasterization  # noqa: E402
from gsplat.cuda import _wrapper as W  # noqa: E402

out = {"gpu": GPU}
snap = torch.load(SNAP, map_location="cpu", weights_only=False)
params = {"means": snap["xyz"].cuda().float(), "quats": snap["rotation"].cuda().float(),
          "log_scales": snap["scaling"].cuda().float(),
          "opacity_logits": snap["opacity"].cuda().float().reshape(-1),
          "sh0": snap["shs"][:, :1].cuda().float(), "shN": snap["shs"][:, 1:].cuda().float()}
degree = int(snap["active_sh_degree"])
seq = np.load(SEQP)
ids = list(dict.fromkeys(int(c) for c in seq.tolist()))[:3]
dataset = G.build_dataset("bicycle")
cam_ds, gt = dataset.get_item(ids[0])
cam = G.camera_dict(cam_ds)
if tuple(gt.shape) == (3, cam["height"], cam["width"]):
    gt = gt.permute(1, 2, 0)
gt = gt.float().contiguous()

N = params["means"].shape[0]
out["n_gaussians"] = N

# ---------- A: covariance convention -----------------------------------------
cov_prism = covariance(params["quats"], params["log_scales"])
tri = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
cov6 = cov_prism[..., tri[0], tri[1]].contiguous()
q_n = F.normalize(params["quats"], dim=-1)
s_e = params["log_scales"].exp()
with torch.no_grad():
    r_a, m2_a, d_a, c_a, comp_a = W.fully_fused_projection(
        params["means"][None], None, q_n[None], s_e[None],
        cam["viewmat"][None], cam["K"][None], cam["width"], cam["height"], eps2d=0.3)
    r_b, m2_b, d_b, c_b, comp_b = W.fully_fused_projection(
        params["means"][None], cov6[None], None, None,
        cam["viewmat"][None], cam["K"][None], cam["width"], cam["height"], eps2d=0.3)
out["covars_convention"] = {
    "radii_equal_fraction": float((r_a == r_b).float().mean()),
    "means2d_maxabs": float((m2_a - m2_b).abs().max()),
    "conics_maxabs": float((c_a - c_b).abs().max()),
    "conics_relative": float((c_a - c_b).abs().max() / c_a.abs().max()),
    "compensations_same": bool(torch.equal(comp_a, comp_b)) if comp_a is not None and comp_b is not None else None,
    "sampled_conic_row0_a": c_a[0, 0].tolist(), "sampled_conic_row0_b": c_b[0, 0].tolist(),
}

# manual check: prism cov vs R diag(s^2) R^T from wrapper's quat->R (via _torch_impl?)
try:
    from gsplat.cuda import _torch_impl as TI
    R = TI._quat_to_rotmat(q_n) if hasattr(TI, "_quat_to_rotmat") else None
    if R is not None:
        cov_ref = (R * s_e.square()[..., None, :]) @ R.transpose(-1, -2)
        out["covars_convention"]["torchimpl_vs_prism_maxabs"] = float((cov_ref - cov_prism).abs().max())
except Exception as e:
    out["covars_convention"]["torchimpl_error"] = repr(e)

# ---------- B: radii / zero-override -----------------------------------------
with torch.no_grad():
    image, st = B.state(params, cam, degree)  # full-param render
    meta = st["meta"]
    radii = meta["radii"][0]                     # [N,2]
    opacity = meta["opacities"][0]               # projected opacities
    cov2d = conic_to_cov(meta["conics"][0])      # [N,2,2]
    diag = cov2d.diagonal(dim1=-2, dim2=-1)
    extend = (2 * torch.log((opacity * 255).clamp_min(1))).sqrt().clamp_max(3.33)
    rr = torch.ceil(extend[:, None] * diag.sqrt()).to(radii.dtype)
    diff = (rr != radii)
    out["radii_formula"] = {
        "components_mismatch": int(diff.sum()),
        "rows_mismatch": int((diff.any(-1)).sum()),
        "rows_total": int(N),
        "opacities_stats": {"min": float(opacity.min()), "median": float(opacity.median()),
                            "max": float(opacity.max())},
        "mismatch_examples": [],
    }
    ex = diff.any(-1).nonzero().flatten()[:8]
    for i in ex.tolist():
        out["radii_formula"]["mismatch_examples"].append({
            "i": int(i), "opacity": float(opacity[i]), "radii_meta": radii[i].tolist(),
            "radii_adapter": rr[i].tolist(), "cov_diag": diag[i].tolist()})
    # re-render using the adapter's draw with z (round trip) on the 256 valid
    valid = st["mask"][0].nonzero().flatten()[:256]
    z = B.features(st, valid)
    zero = B.draw(st, valid, z)
    out["zero_override"] = {
        "maxabs_drawable": float((image - zero).abs().max()),
        "rows_256_of_N": 256,
        "radii_256_mismatch": int((rr[valid] != radii[valid]).any(-1).sum()),
        "target_tolerance": 2e-6,
    }

# ---------- C: SH color / gradient paths --------------------------------------
mask = st["mask"][0]
colors_adapter = B.colors_for(params, cam, degree, mask)
from gsplat.cuda._wrapper import spherical_harmonics as SH
colors_wrap = (SH(degree, params["means"], cam["viewmat"][None],
                  torch.cat((params["sh0"], params["shN"]), 1), masks=(meta["radii"] > 0).all(dim=-1)) + .5).clamp_min(0)
internal_masks = (meta["radii"] > 0).all(dim=-1)
colors_adapter_all = (SH(degree, params["means"], cam["viewmat"][None],
                         torch.cat((params["sh0"], params["shN"]), 1), masks=internal_masks) + .5).clamp_min(0)
out["sh_colors"] = {
    "adapter_vs_wrapper_maxabs": float((colors_adapter - colors_wrap).abs().max()),
    "any_vs_all_mask_rows_diff": int((mask != internal_masks[0]).sum()),
    "visible_color_maxabs": float((colors_adapter[0][mask] - colors_wrap[0][mask]).abs().max()),
}


def grads_through(fn):
    pp = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
    img = fn(pp)
    loss = G.fused_loss(img, gt)
    g = torch.autograd.grad(loss, [pp["means"], pp["quats"], pp["log_scales"],
                                   pp["opacity_logits"], pp["sh0"], pp["shN"]],
                            allow_unused=True)
    return img.detach(), loss.detach(), [torch.zeros_like(pp["means"]) if x is None else x.detach() for x in g]


def path_internal(pp):
    img, _meta = B.baseline(pp, cam, degree)
    return img


def path_draw(pp):
    img, st = B.state(pp, cam, degree)
    return B.draw(st)


def path_rgb(pp):
    from prism.gsplat_backend import colors_for as CF
    img, st = B.state(pp, cam, degree)
    m = st["mask"][0]
    cols = CF(pp, cam, degree, m)
    _r, _radii, _meta = rasterization(
        means=pp["means"], quats=F.normalize(pp["quats"], dim=-1), scales=pp["log_scales"].exp(),
        opacities=pp["opacity_logits"].sigmoid(), colors=cols,
        viewmats=cam["viewmat"][None], Ks=cam["K"][None],
        width=cam["width"], height=cam["height"], sh_degree=None,
        eps2d=.3, radius_clip=0., tile_size=16, packed=False, absgrad=True,
        render_mode="RGB", rasterize_mode="classic")
    return _r[0].clamp(0, 1)


names = ["means", "quats", "log_scales", "opacity_logits", "sh0", "shN"]
img_i, loss_i, g_i = grads_through(path_internal)
img_d, loss_d, g_d = grads_through(path_draw)
img_r, loss_r, g_r = grads_through(path_rgb)
cmp = {}
for name, a, b, c in zip(names, g_i, g_d, g_r):
    d1 = (a.double() - b.double())
    d2 = (a.double() - c.double())
    cmp[name] = {
        "internal_vs_draw": {"maxabs": float(d1.abs().max()),
                             "rel_l2": float(d1.norm() / a.double().norm().clamp_min(1e-20))},
        "internal_vs_rgb": {"maxabs": float(d2.abs().max()),
                            "rel_l2": float(d2.norm() / a.double().norm().clamp_min(1e-20))},
        "internal_norm": float(a.double().norm()), "internal_maxabs": float(a.abs().max()),
    }
out["grad_paths"] = {"losses": {"internal": float(loss_i), "draw": float(loss_d), "rgb": float(loss_r)},
                     "images_bitwise_equal_draw": float((img_i - img_d).abs().max()),
                     "images_bitwise_equal_rgb": float((img_i - img_r).abs().max()),
                     "groups": cmp}
# where is the sh0 diff concentrated (internal vs draw)?
d_sh0 = (g_i[4] - g_d[4])
idx = d_sh0.abs().reshape(d_sh0.shape[0], -1).max(-1).values.topk(5).indices
out["sh0_diff_top_rows"] = [{"row": int(i), "internal": g_i[4][i].flatten()[:4].tolist(),
                             "draw": g_d[4][i].flatten()[:4].tolist()} for i in idx.tolist()]
with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str)[:6000])
