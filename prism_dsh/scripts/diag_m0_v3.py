"""Diagnose BLOCKED_M0 sub-checks, v3. Fixes: [B,C] dims for projection op,
2-D masks for SH op, valid-row-only conic inversion, numpy histogram."""
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
OUT = "/root/epic05_prism/outputs/diag_m0_v3_bicycle_5000.json"

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
cam_ids = list(dict.fromkeys(int(c) for c in seq.tolist()))[:3]
dataset = G.build_dataset("bicycle")
cam_ds, gt = dataset.get_item(cam_ids[0])
cam = G.camera_dict(cam_ds)
if tuple(gt.shape) == (3, cam["height"], cam["width"]):
    gt = gt.permute(1, 2, 0)
gt = gt.float().contiguous()
N = params["means"].shape[0]
cov_prism = covariance(params["quats"], params["log_scales"])
tri = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
cov6 = cov_prism[..., tri[0], tri[1]]

# ---------- A: wrapper-level projection, quats vs covars ---------------------
try:
    q_n = F.normalize(params["quats"], dim=-1)
    s_e = params["log_scales"].exp()
    op = params["opacity_logits"].sigmoid()
    with torch.no_grad():
        r_a, m2_a, d_a, c_a, _ = W.fully_fused_projection(
            params["means"][None], None, q_n[None], s_e[None],
            cam["viewmat"][None], cam["K"][None], cam["width"], cam["height"],
            eps2d=0.3, opacities=op[None])
        r_b, m2_b, d_b, c_b, _ = W.fully_fused_projection(
            params["means"][None], cov6[None], None, None,
            cam["viewmat"][None], cam["K"][None], cam["width"], cam["height"],
            eps2d=0.3, opacities=op[None])
    va = (r_a[0] > 0).any(-1)
    out["A_wrapper_proj"] = {
        "radii_mismatch_rows": int(((r_a != r_b).any(-1))[0].sum()),
        "means2d_maxabs_valid": float((m2_a[0] - m2_b[0])[va].abs().max()),
        "conics_maxabs_valid": float((c_a[0] - c_b[0])[va].abs().max()),
        "conics_rel_valid": float(((c_a[0] - c_b[0])[va].abs() / c_a[0][va].abs().clamp_min(1e-20)).max()),
        "depths_maxabs_valid": float((d_a[0] - d_b[0])[va].abs().max()),
    }
except Exception:
    out["A_wrapper_proj"] = {"error": traceback.format_exc()}

# ---------- A2: orchestrator-level quats vs covars render --------------------
try:
    with torch.no_grad():
        img_q, meta_q = B.baseline(params, cam, degree)
        img_c, meta_c = B.baseline(params, cam, degree, covars=cov_prism)
    mq, mc = meta_q, meta_c
    rq, rc = mq["radii"][0], mc["radii"][0]
    valid = (rq > 0).any(-1) & (rc > 0).any(-1)
    cov_q = conic_to_cov(mq["conics"][0][valid])
    cov_c = conic_to_cov(mc["conics"][0][valid])
    dvv = (cov_q - cov_c).abs() / cov_q.abs().clamp_min(1e-12)
    dv = (rq - rc)[valid]
    vals, counts = np.unique(dv.cpu().numpy(), return_counts=True)
    out["A2_orchestrator"] = {
        "image_maxabs": float((img_q - img_c).abs().max()),
        "radii_mismatch_rows_all": int(((rq != rc).any(-1)).sum()),
        "radii_mismatch_rows_valid": int(((rq[valid] != rc[valid]).any(-1)).sum()),
        "valid_rows": int(valid.sum()),
        "radii_diff_values": {str(int(v)): int(c) for v, c in zip(vals, counts)},
        "means2d_maxabs_valid": float((mq["means2d"][0] - mc["means2d"][0])[valid].abs().max()),
        "conics_rel_max_valid": float((((mq["conics"][0] - mc["conics"][0])[valid]).abs()
                                       / mq["conics"][0][valid].abs().clamp_min(1e-20)).max()),
        "cov2d_rel_max_valid": float(dvv.max()),
        "opacities_maxabs_valid": float((mq["opacities"][0] - mc["opacities"][0])[valid].abs().max()),
        "depths_maxabs_valid": float((mq["depths"][0] - mc["depths"][0])[valid].abs().max()),
    }
    ex = ((rq != rc).any(-1) & valid).nonzero().flatten()[:6]
    out["A2_orchestrator"]["radii_examples"] = [
        {"i": int(i), "r_quat": rq[i].tolist(), "r_cov": rc[i].tolist(),
         "opacity": float(mq["opacities"][0, i])} for i in ex.tolist()]
except Exception:
    out["A2_orchestrator"] = {"error": traceback.format_exc()}

# ---------- B: radii formula + zero-override ---------------------------------
try:
    with torch.no_grad():
        image, st = B.state(params, cam, degree)
        meta = st["meta"]
        radii = meta["radii"][0]
        opacity = meta["opacities"][0]
        rows_ok = (radii > 0).any(-1)
        conics = meta["conics"][0]
        zero_conic_rows = int((conics.abs().sum(-1) == 0).sum())
        cov2d = torch.zeros(N, 2, 2, device=conics.device, dtype=conics.dtype)
        cov2d[rows_ok] = conic_to_cov(conics[rows_ok])
        diag = cov2d.diagonal(dim1=-2, dim2=-1)
        extend = (2 * torch.log((opacity * 255).clamp_min(1))).sqrt().clamp_max(3.33)
        rr = torch.ceil(extend[:, None] * diag.sqrt()).to(radii.dtype)
        diff = (rr != radii) & rows_ok[:, None]
        out["B_radii_formula"] = {
            "rows_total": int(N), "rows_ok": int(rows_ok.sum()),
            "zero_conic_rows": zero_conic_rows,
            "rows_mismatch": int(diff.any(-1).sum()),
            "opacities_min_nonzero": float(opacity[rows_ok].min()),
            "opacities_median": float(opacity[rows_ok].median()),
            "mismatch_examples": [],
        }
        ex = diff.any(-1).nonzero().flatten()[:8]
        for i in ex.tolist():
            out["B_radii_formula"]["mismatch_examples"].append({
                "i": int(i), "opacity": float(opacity[i]), "radii_meta": radii[i].tolist(),
                "radii_adapter": rr[i].tolist(), "cov_diag": diag[i].tolist(),
                "raw_extend": float(extend[i])})
        valid = st["mask"][0].nonzero().flatten()[:256]
        z = B.features(st, valid)
        uv_rt, cov_rt = screen_unpack(z)
        uv_err = float((uv_rt - meta["means2d"][0, valid]).abs().max())
        cov_err = float((cov_rt - cov2d[valid]).abs().max())
        zero = B.draw(st, valid, z)
        d = (image - zero).abs().amax(-1)
        nz = (d > 2e-6).nonzero()
        out["B_zero_override"] = {
            "maxabs_drawable": float(d.max()),
            "uv_roundtrip_maxabs": uv_err,
            "cov_roundtrip_maxabs": cov_err,
            "radii_256_mismatch": int((rr[valid] != radii[valid]).any(-1).sum()),
            "moved_pixels_gt_2e6": int(nz.shape[0]),
            "target_tolerance": 2e-6,
        }
        # is the moved-pixel difference explained by radius bbox flips?
        diff_rows = (rr != radii).any(-1).nonzero().flatten()
        nuv, ncov = screen_unpack(B.features(st, diff_rows))
        extend_d = (2 * torch.log((opacity[diff_rows] * 255).clamp_min(1))).sqrt().clamp_max(3.33)
        rr2 = torch.ceil(extend_d[:, None] * ncov.diagonal(dim1=-2, dim2=-1).sqrt()).to(radii.dtype)
        out["B_zero_override"]["allN_radius_rows_flip"] = int(diff_rows.numel())
        out["B_zero_override"]["flip_values"] = {str(int(v)): int(c) for v, c in
                                                 zip(*np.unique((rr[diff_rows] - radii[diff_rows]).cpu().numpy(),
                                                                return_counts=True))} if diff_rows.numel() else {}
except Exception:
    out["B_radii_formula"] = {"error": traceback.format_exc()}

# ---------- C: SH colors + gradient paths ------------------------------------
mask_any = None
mask_all = None
try:
    mask_any = st["mask"]                       # [1,N] any(-1)
    mask_all = (meta["radii"] > 0).all(-1)      # [1,N]
    sh_all = torch.cat((params["sh0"], params["shN"]), 1)
    with torch.no_grad():
        col_any = (SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_any) + .5).clamp_min(0)
        col_all = (SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_all) + .5).clamp_min(0)
    out["C_masks"] = {
        "rows_any_not_all": int((mask_any & ~mask_all).sum()),
        "colors_any_vs_all_maxabs": float((col_any - col_all).abs().max()),
        "rows_with_color_diff": int(((col_any - col_all).abs().amax(-1) > 0).sum()),
    }
    g_a = []
    for msk in (mask_any, mask_all):
        g0 = params["sh0"].detach().clone().requires_grad_()
        shx = torch.cat((g0, params["shN"]), 1)
        cc = (SH(degree, params["means"], cam["viewmat"][None], shx, masks=msk) + .5).clamp_min(0)
        g_a.append(torch.autograd.grad(cc.sum(), g0)[0].detach())
    dga = (g_a[0].double() - g_a[1].double())
    out["C_masks"]["sh0_grad_any_vs_all_maxabs"] = float(dga.abs().max())
except Exception:
    out["C_masks"] = {"error": traceback.format_exc()}


def grads_through(fn):
    pp = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
    img = fn(pp)
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
    img_i, loss_i, g_i = grads_through(lambda pp: B.baseline(pp, cam, degree)[0])

    def draw_route(pp):
        _img, stt = B.state(pp, cam, degree)
        return B.draw(stt)

    def rgb_any(pp):
        shx = torch.cat((pp["sh0"], pp["shN"]), 1)
        return rgb_render(pp, (SH(degree, pp["means"], cam["viewmat"][None], shx, masks=mask_any) + .5).clamp_min(0))

    def rgb_all(pp):
        shx = torch.cat((pp["sh0"], pp["shN"]), 1)
        return rgb_render(pp, (SH(degree, pp["means"], cam["viewmat"][None], shx, masks=mask_all) + .5).clamp_min(0))

    img_d, loss_d, g_d = grads_through(draw_route)
    img_ra, loss_ra, g_ra = grads_through(rgb_any)
    img_rl, loss_rl, g_rl = grads_through(rgb_all)

    cmp = {}
    for name, a, b, c, e in zip(names, g_i, g_d, g_ra, g_rl):
        d1 = (a.double() - b.double())
        d2 = (a.double() - c.double())
        d3 = (a.double() - e.double())
        cmp[name] = {
            "vs_draw_rel_l2": float(d1.norm() / a.double().norm().clamp_min(1e-20)),
            "vs_rgb_any_rel_l2": float(d2.norm() / a.double().norm().clamp_min(1e-20)),
            "vs_rgb_all_rel_l2": float(d3.norm() / a.double().norm().clamp_min(1e-20)),
            "vs_rgb_all_maxabs": float(d3.abs().max()),
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
print(json.dumps(out, indent=2, default=str)[:9000])
