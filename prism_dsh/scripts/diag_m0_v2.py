"""Diagnose the three BLOCKED_M0 sub-checks (covars re-render, zero-override,
sh0/shN gradient mismatch) on the pinned custom gsplat build. Read-only."""
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
OUT = "/root/epic05_prism/outputs/diag_m0_v2_bicycle_5000.json"

GPU = int(os.environ.get("PRISM_GPU", "2"))
sys.path.insert(0, GX)
sys.path.insert(0, BUNDLE)
import gx_common as G  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
trainer, ident = G.boot_gmain(GPU)
from prism import gsplat_backend as B  # noqa: E402
from prism.geometry import covariance, conic_to_cov, cov_to_conic, screen_pack, screen_unpack  # noqa: E402
from gsplat import rasterization  # noqa: E402
from gsplat.cuda import _wrapper as W  # noqa: E402
from gsplat.cuda._wrapper import spherical_harmonics as SH  # noqa: E402

out = {"gpu": GPU}
snap = torch.load(SNAP, map_location="cpu", weights_only=False)
params = {"means": snap["xyz"].cuda().float(), "quats": snap["rotation"].cuda().float(),
          "log_scales": snap["scaling"].cuda().float(),
          "opacity_logits": snap["opacity"].cuda().float().reshape(-1),
          "sh0": snap["shs"][:, :1].cuda().float(), "shN": snap["shs"][:, 1:].cuda().float()}
degree = int(snap["active_sh_degree"])
seq = np.load(SEQP)
ids_all = list(dict.fromkeys(int(c) for c in seq.tolist()))
cam_ids = ids_all[:3]
dataset = G.build_dataset("bicycle")
cam_ds, gt = dataset.get_item(cam_ids[0])
cam = G.camera_dict(cam_ds)
if tuple(gt.shape) == (3, cam["height"], cam["width"]):
    gt = gt.permute(1, 2, 0)
gt = gt.float().contiguous()
N = params["means"].shape[0]
out["n_gaussians"] = N
out["quat_norm_stats"] = {"mean": float(params["quats"].norm(dim=-1).mean()),
                          "min": float(params["quats"].norm(dim=-1).min())}

cov_prism = covariance(params["quats"], params["log_scales"])

# ---------- A: wrapper-level quats vs covars projection ----------------------
try:
    tri = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    cov6 = cov_prism[..., tri[0], tri[1]].contiguous()
    q_n = F.normalize(params["quats"], dim=-1)
    s_e = params["log_scales"].exp()
    op = params["opacity_logits"].sigmoid()
    with torch.no_grad():
        r_a, m2_a, d_a, c_a, comp_a = W.fully_fused_projection(
            params["means"][None, None], None, q_n[None, None], s_e[None, None],
            cam["viewmat"][None, None], cam["K"][None, None], cam["width"], cam["height"],
            eps2d=0.3, opacities=op[None, None])
        r_b, m2_b, d_b, c_b, comp_b = W.fully_fused_projection(
            params["means"][None, None], cov6[None, None], None, None,
            cam["viewmat"][None, None], cam["K"][None, None], cam["width"], cam["height"],
            eps2d=0.3, opacities=op[None, None])
    out["A_wrapper_proj"] = {
        "radii_mismatch_rows": int((r_a[0, 0] != r_b[0, 0]).any(-1).sum()),
        "means2d_maxabs": float((m2_a - m2_b).abs().max()),
        "conics_maxabs": float((c_a - c_b).abs().max()),
        "conics_rel": float((c_a - c_b).abs().max() / c_a.abs().max()),
        "depths_maxabs": float((d_a - d_b).abs().max()),
    }
except Exception:
    out["A_wrapper_proj"] = {"error": traceback.format_exc()}

try:
    from gsplat.cuda import _torch_impl as TI
    if hasattr(TI, "_quat_to_rotmat"):
        R = TI._quat_to_rotmat(params["quats"])
        cov_ref = (R * params["log_scales"].exp().square()[..., None, :]) @ R.transpose(-1, -2)
        out["A_torchimpl_vs_prism_cov_maxabs"] = float((cov_ref - cov_prism).abs().max())
    else:
        from gsplat.cuda._math import _quat_to_rotmat as q2r
        R = q2r(params["quats"])
        cov_ref = (R * params["log_scales"].exp().square()[..., None, :]) @ R.transpose(-1, -2)
        out["A_torchimpl_vs_prism_cov_maxabs"] = float((cov_ref - cov_prism).abs().max())
except Exception:
    out["A_torchimpl_vs_prism_cov_maxabs"] = traceback.format_exc()

# ---------- A2: orchestrator-level quats vs covars render --------------------
try:
    with torch.no_grad():
        img_q, meta_q = B.baseline(params, cam, degree)
        img_c, meta_c = B.baseline(params, cam, degree, covars=cov_prism)
    mq, mc = meta_q, meta_c
    rq, rc = mq["radii"][0], mc["radii"][0]
    rowdiff = (rq != rc).any(-1)
    ex = rowdiff.nonzero().flatten()[:6]
    out["A2_orchestrator"] = {
        "image_maxabs": float((img_q - img_c).abs().max()),
        "radii_mismatch_rows": int(rowdiff.sum()),
        "means2d_maxabs": float((mq["means2d"] - mc["means2d"]).abs().max()),
        "conics_maxabs": float((mq["conics"] - mc["conics"]).abs().max()),
        "opacities_maxabs": float((mq["opacities"] - mc["opacities"]).abs().max()),
        "depths_maxabs": float((mq["depths"] - mc["depths"]).abs().max()),
        "radii_examples": [{"i": int(i), "r_quat": rq[i].tolist(), "r_cov": rc[i].tolist()}
                           for i in ex.tolist()],
        "radii_diff_abs_hist": {str(k): int(v) for k, v in
                                zip(*[(int(d), int((rq - rc).abs() == d).sum())
                                      for d in (rq - rc).abs().unique()[:12]])},
    }
except Exception:
    out["A2_orchestrator"] = {"error": traceback.format_exc()}

# ---------- B: radii formula + zero-override ---------------------------------
try:
    with torch.no_grad():
        image, st = B.state(params, cam, degree)
        meta = st["meta"]
        radii = meta["radii"][0]
        opacity = meta["opacities"][0]
        cov2d = conic_to_cov(meta["conics"][0])
        diag = cov2d.diagonal(dim1=-2, dim2=-1)
        extend = (2 * torch.log((opacity * 255).clamp_min(1))).sqrt().clamp_max(3.33)
        rr = torch.ceil(extend[:, None] * diag.sqrt()).to(radii.dtype)
        diff = (rr != radii)
        out["B_radii_formula"] = {
            "rows_mismatch": int(diff.any(-1).sum()),
            "rows_total": int(N),
            "opacities_min": float(opacity.min()), "opacities_median": float(opacity.median()),
            "mismatch_examples": [],
        }
        ex = diff.any(-1).nonzero().flatten()[:8]
        for i in ex.tolist():
            out["B_radii_formula"]["mismatch_examples"].append({
                "i": int(i), "opacity": float(opacity[i]), "radii_meta": radii[i].tolist(),
                "radii_adapter": rr[i].tolist(), "cov_diag": diag[i].tolist(),
                "raw_extend_arg": float(extend[i])})
        valid = st["mask"][0].nonzero().flatten()[:256]
        z = B.features(st, valid)
        uv_rt, cov_rt = screen_unpack(z)
        uv_err = float((uv_rt - meta["means2d"][0, valid]).abs().max())
        cov_err = float((cov_rt - cov2d[valid]).abs().max())
        x0, _ = screen_unpack(B.features(st, valid))
        zero = B.draw(st, valid, z)
        out["B_zero_override"] = {
            "maxabs_drawable": float((image - zero).abs().max()),
            "uv_roundtrip_maxabs": uv_err,
            "cov_roundtrip_maxabs": cov_err,
            "radii_256_mismatch": int((rr[valid] != radii[valid]).any(-1).sum()),
            "target_tolerance": 2e-6,
        }
        # which pixels moved?
        d = (image - zero).abs().amax(-1)
        nz = (d > 2e-6).nonzero()
        out["B_zero_override"]["moved_pixels"] = int(nz.shape[0])
        if nz.shape[0]:
            out["B_zero_override"]["worst_pixel"] = nz[d[nz[:, 0], nz[:, 1]].argmax()].tolist()
except Exception:
    out["B_radii_formula"] = {"error": traceback.format_exc()}

# ---------- C: SH colors + gradient paths ------------------------------------
try:
    mask_any = st["mask"][0]
    mask_all = (meta["radii"] > 0).all(dim=-1)[0]
    sh_all = torch.cat((params["sh0"], params["shN"]), 1)
    with torch.no_grad():
        col_any = (SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_any) + .5).clamp_min(0)
        col_all = (SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_all) + .5).clamp_min(0)
    out["C_masks"] = {
        "rows_any_not_all": int((mask_any & ~mask_all).sum()),
        "colors_any_vs_all_maxabs": float((col_any - col_all).abs().max()),
        "rows_with_color_diff": int(((col_any - col_all).abs().amax(-1) > 0).sum()),
    }
    sh0g = params["sh0"].detach().clone().requires_grad_()
    sh_ga = torch.cat((sh0g, params["shN"]), 1)
    c_a = (SH(degree, params["means"], cam["viewmat"][None], sh_ga, masks=mask_any) + .5).clamp_min(0)
    g_any = torch.autograd.grad(c_a.sum(), sh0g)[0]
    sh0g2 = params["sh0"].detach().clone().requires_grad_()
    sh_ga2 = torch.cat((sh0g2, params["shN"]), 1)
    c_b = (SH(degree, params["means"], cam["viewmat"][None], sh_ga2, masks=mask_all) + .5).clamp_min(0)
    g_all = torch.autograd.grad(c_b.sum(), sh0g2)[0]
    dga = (g_any.double() - g_all.double())
    out["C_masks"]["sh0_grad_any_vs_all_maxabs"] = float(dga.abs().max())
except Exception:
    out["C_masks"] = {"error": traceback.format_exc()}


def grads_through(fn, cols=None):
    pp = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
    img = fn(pp, cols)
    loss = G.fused_loss(img, gt)
    g = torch.autograd.grad(loss, [pp["means"], pp["quats"], pp["log_scales"],
                                   pp["opacity_logits"], pp["sh0"], pp["shN"]],
                            allow_unused=True)
    return img.detach(), loss.detach(), [torch.zeros_like(pp["means"]) if x is None else x.detach()
                                         for x in g]


def rgb_render(pp, cols):
    _r, _a, _m = rasterization(
        means=pp["means"], quats=F.normalize(pp["quats"], dim=-1), scales=pp["log_scales"].exp(),
        opacities=pp["opacity_logits"].sigmoid(), colors=cols,
        viewmats=cam["viewmat"][None], Ks=cam["K"][None],
        width=cam["width"], height=cam["height"], sh_degree=None,
        eps2d=.3, radius_clip=0., tile_size=16, packed=False, absgrad=True,
        render_mode="RGB", rasterize_mode="classic")
    return _r[0].clamp(0, 1)


try:
    names = ["means", "quats", "log_scales", "opacity_logits", "sh0", "shN"]
    img_i, loss_i, g_i = grads_through(lambda pp, c: B.baseline(pp, cam, degree)[0])

    def draw_route(pp, cols):
        _img, stt = B.state(pp, cam, degree)
        return B.draw(stt)
    img_d, loss_d, g_d = grads_through(draw_route)

    def rgb_any(pp, cols):
        shx = torch.cat((pp["sh0"], pp["shN"]), 1)
        cc = (SH(degree, pp["means"], cam["viewmat"][None], shx, masks=mask_any) + .5).clamp_min(0)
        return rgb_render(pp, cc)

    def rgb_all(pp, cols):
        shx = torch.cat((pp["sh0"], pp["shN"]), 1)
        cc = (SH(degree, pp["means"], cam["viewmat"][None], shx, masks=mask_all) + .5).clamp_min(0)
        return rgb_render(pp, cc)
    img_ra, loss_ra, g_ra = grads_through(rgb_any)
    img_rl, loss_rl, g_rl = grads_through(rgb_all)

    cmp = {}
    for name, a, b, c, e in zip(names, g_i, g_d, g_ra, g_rl):
        d1 = (a.double() - b.double())
        d2 = (a.double() - c.double())
        d3 = (a.double() - e.double())
        cmp[name] = {
            "internal_vs_draw_rel_l2": float(d1.norm() / a.double().norm().clamp_min(1e-20)),
            "internal_vs_rgb_any_rel_l2": float(d2.norm() / a.double().norm().clamp_min(1e-20)),
            "internal_vs_rgb_all_rel_l2": float(d3.norm() / a.double().norm().clamp_min(1e-20)),
            "internal_vs_rgb_any_maxabs": float(d2.abs().max()),
            "internal_norm": float(a.double().norm()),
        }
    out["C_grad_paths"] = {
        "losses": {"internal": float(loss_i), "draw": float(loss_d),
                   "rgb_any": float(loss_ra), "rgb_all": float(loss_rl)},
        "img_internal_vs_draw": float((img_i - img_d).abs().max()),
        "img_internal_vs_rgb_any": float((img_i - img_ra).abs().max()),
        "img_internal_vs_rgb_all": float((img_i - img_rl).abs().max()),
        "groups": cmp,
    }
except Exception:
    out["C_grad_paths"] = {"error": traceback.format_exc()}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str)[:8000])
