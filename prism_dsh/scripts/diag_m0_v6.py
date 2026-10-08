"""v6: (1) classify sh-grad differing entries (masked / near-zero-color / contributing
rows), (2) fixed zero-override causality with a general rebuild, (3) SH-bwd shape
variants with valid mask shapes, (4) conic-patch causality for the covars route."""
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
OUT = "/root/epic05_prism/outputs/diag_m0_v6_bicycle_5000.json"

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
K = sh_all.shape[1]


def rel(a, b):
    a = a.double(); b = b.double()
    return {"rel_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
            "maxabs": float((a - b).abs().max()), "norm": float(a.norm())}


def rebuild_full(uv, co, radii, colors, depth, opac, tile=16):
    if uv.dim() == 2:
        uv = uv[None]
    if co.dim() == 2:
        co = co[None]
    if radii.dim() == 2:
        radii = radii[None]
    if depth.dim() == 1:
        depth = depth[None]
    if opac.dim() == 1:
        opac = opac[None]
    tw = math.ceil(cam["width"] / tile); th = math.ceil(cam["height"] / tile)
    _, isect, flat = isect_tiles(uv, radii, depth, tile, tw, th, packed=False)
    off = isect_offset_encode(isect, 1, tw, th)
    img, _ = rasterize_to_pixels(uv, co, colors, opac, cam["width"], cam["height"],
                                 tile, off, flat, packed=False, absgrad=False)
    return img[0].clamp(0, 1), flat


with torch.no_grad():
    img_q, meta_q = B.baseline(params, cam, degree)
    img_c, meta_c = B.baseline(params, cam, degree,
                               covars=covariance(params["quats"], params["log_scales"]))
mask_all = (meta_q["radii"] > 0).all(-1)          # [1,N]
c_draw = (SH(degree, params["means"], cam["viewmat"][None], sh_all, masks=mask_all) + .5).clamp_min(0)
c_draw1 = c_draw[0] if c_draw.dim() == 3 else c_draw   # [N,3]

# flat isect list of the state (contributing gaussian rows)
with torch.no_grad():
    tw = math.ceil(cam["width"] / 16); th = math.ceil(cam["height"] / 16)
    _, isect_q, flat_q = isect_tiles(meta_q["means2d"], meta_q["radii"], meta_q["depths"], 16, tw, th, packed=False)
in_isect = torch.zeros(N, dtype=torch.bool, device="cuda")
in_isect[flat_q.flatten().long()] = True

# ---------- 1/2/3: sh-grad differing entries ---------------------------------
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

    g_i = grads(internal)
    g_r = grads(rgb_route)
    gi_all = torch.cat((g_i[4], g_i[5]), 1)   # [N,K,3]
    gr_all = torch.cat((g_r[4], g_r[5]), 1)
    d = (gi_all - gr_all).abs()
    # entry classes
    m_leaf = mask_all[0]                                   # [N] visible rows
    c0 = (c_draw1 == 0.0).all(-1)                          # row whose draw color is exactly 0
    near = (c_draw1.abs() < 1e-6).any(-1)
    ent = d > 0
    n_ent = int(ent.sum())
    row_has = ent.any(-1)
    out["1_entry_classes"] = {
        "n_diff_entries": n_ent,
        "n_rows_any_diff": int(row_has.sum()),
        "diff_entries_on_masked_rows": int((ent & (~m_leaf)[:, None, None]).sum()),
        "diff_entries_on_visible_rows": int((ent & m_leaf[:, None, None]).sum()),
        "visible_rows_total": int(m_leaf.sum()),
        "masked_rows_total": int((~m_leaf).sum()),
        "diff_entries_rows_in_isect": int((ent.any(-1) & in_isect).sum()),
        "diff_rows_in_isect": int((row_has & in_isect).sum()),
        "diff_rows_not_in_isect": int((row_has & ~in_isect).sum()),
        "diff_entries_channel_color_zero": int((ent & (c_draw1 == 0.0)[:, None, :]).sum()),
        "diff_entries_channel_color_lt1e-6": int((ent & (c_draw1.abs() < 1e-6)[:, None, :]).sum()),
        "rows_all_colors_zero": int(c0.sum()), "rows_near_zero_color": int(near.sum()),
        "total_entries": int(ent.numel()),
        "rgb_grad_norm": float(gr_all.double().norm()), "int_grad_norm": float(gi_all.double().norm()),
    }
    # relation internal vs draw on differing entries
    di = gi_all[ent].double(); dr = gr_all[ent].double()
    out["1_entry_classes"]["diff_entries_internal_zero"] = int((di == 0).sum())
    out["1_entry_classes"]["diff_entries_draw_zero"] = int((dr == 0).sum())
    # examples: top 8 by |draw|
    idx = gr_all.abs().flatten().topk(8).indices
    ex = []
    for f in idx.tolist():
        r = f // (K * 3); k = (f % (K * 3)) // 3; ch = f % 3
        ex.append({"flat": f, "row": int(r), "k": int(k), "ch": int(ch),
                   "internal": float(gi_all[r, k, ch]), "draw": float(gr_all[r, k, ch]),
                   "color_draw": float(c_draw1[r, ch]), "visible": bool(m_leaf[r]),
                   "in_isect": bool(in_isect[r]),
                   "radii": meta_q["radii"][0, r].tolist(),
                   "opacity": float(meta_q["opacities"][0, r] if meta_q["opacities"].dim() == 2
                                     else meta_q["opacities"][r])})
    out["2_top8_entries"] = ex
    # rows with the largest row-level diff
    row_dmax = d.amax(dim=(1, 2))
    top_rows = row_dmax.topk(6).indices.tolist()
    out["3_top_rows"] = [{
        "row": int(r), "row_dmax": float(row_dmax[r]),
        "visible": bool(m_leaf[r]), "in_isect": bool(in_isect[r]),
        "n_diff_entries": int((d[r] > 0).sum()),
        "internal_row_norm": float(gi_all[r].double().norm()),
        "draw_row_norm": float(gr_all[r].double().norm()),
        "color_draw_row": [float(x) for x in c_draw1[r].tolist()],
        "radii": meta_q["radii"][0, r].tolist(),
        "opacity": float(meta_q["opacities"].reshape(-1)[r]),
    } for r in top_rows]
    del g_i, g_r, gi_all, gr_all, d, row_dmax
except Exception:
    out["1_entry_classes"] = {"error": traceback.format_exc()}

# ---------- 4: zero-override causality (fixed) --------------------------------
try:
    image, st = B.state(params, cam, degree)
    meta = st["meta"]
    radii = meta["radii"][0]
    rows_ok = (radii > 0).any(-1)
    cov2d = torch.zeros(N, 2, 2, device="cuda")
    cov2d[rows_ok] = conic_to_cov(meta["conics"][0][rows_ok])
    zfull = screen_pack(meta["means2d"][0], cov2d)
    uv_o, cov_o = screen_unpack(zfull)
    opac = meta["opacities"]
    if opac.ndim == 2:
        opac = opac[0]
    extend = (2 * torch.log((opac * 255).clamp_min(1))).sqrt().clamp_max(3.33)
    rr = torch.ceil(extend[:, None] * cov_o.diagonal(dim1=-2, dim2=-1).sqrt()).to(radii.dtype)
    inside = ((uv_o[:, 0] + rr[:, 0] > 0) & (uv_o[:, 0] - rr[:, 0] < cam["width"]) &
              (uv_o[:, 1] + rr[:, 1] > 0) & (uv_o[:, 1] - rr[:, 1] < cam["height"]))
    rr = torch.where(inside[:, None], rr, torch.zeros_like(rr))
    flip2 = (rr != radii).any(-1)
    fl = flip2.nonzero().flatten()
    # M0 target rows: first 256 visible rows
    sel = rows_ok.nonzero().flatten()[:256]
    sel_mask = torch.zeros(N, dtype=torch.bool, device="cuda"); sel_mask[sel] = True
    co_re = cov_to_conic(cov_o)
    colors = st["colors"]
    opac_t = meta["opacities"] if meta["opacities"].ndim == 2 else meta["opacities"][None]
    img_base, _ = rebuild_full(meta["means2d"], meta["conics"], radii[None], colors, meta["depths"], opac_t)
    # variant A: full override (as M0 draw does)
    uv_a = meta["means2d"].clone(); co_a = meta["conics"].clone(); rr_a = radii.clone()
    uv_a[0, sel] = uv_o[sel]; co_a[0, sel] = co_re[sel]; rr_a[sel] = rr[sel]
    img_A, _ = rebuild_full(uv_a, co_a, rr_a[None], colors, meta["depths"], opac_t)
    # variant B: override everything except radii (keep meta radii)
    img_B, _ = rebuild_full(uv_a, co_a, radii[None], colors, meta["depths"], opac_t)
    # variant C: only uv replaced
    img_C, _ = rebuild_full(uv_a, meta["conics"], radii[None], colors, meta["depths"], opac_t)
    # variant D: only conics replaced
    img_D, _ = rebuild_full(meta["means2d"], co_a, radii[None], colors, meta["depths"], opac_t)
    tie = []
    for i in fl[:12].tolist():
        p_a = (extend[i].double() * cov2d[i].diagonal().double().sqrt())
        p_b = (extend[i].double() * cov_o[i].diagonal().double().sqrt())
        tie.append({"row": int(i), "sel": bool(sel_mask[i]),
                    "p_orig": [float(x) for x in p_a.tolist()],
                    "p_roundtrip": [float(x) for x in p_b.tolist()],
                    "dist_to_int_p_orig": [float(min(abs(x - math.floor(x)), abs(x - math.ceil(x)))) for x in p_a.tolist()],
                    "dist_to_int_p_rt": [float(min(abs(x - math.floor(x)), abs(x - math.ceil(x)))) for x in p_b.tolist()],
                    "radii_meta": radii[i].tolist(), "radii_rt": rr[i].tolist()})
    out["4_zero_override"] = {
        "flip_rows_allN": int(flip2.sum()),
        "flip_rows_within_sel256": int((flip2 & sel_mask).sum()),
        "flip_values": {str(int(v)): int(c) for v, c in
                        zip(*np.unique((rr[flip2] - radii[flip2]).cpu().numpy(), return_counts=True))},
        "sanity_base_vs_orig": float((img_base - image).abs().max()),
        "A_full_override_vs_orig": float((img_A - image).abs().max()),
        "B_radii_kept_vs_orig": float((img_B - image).abs().max()),
        "C_only_uv_vs_orig": float((img_C - image).abs().max()),
        "D_only_conic_vs_orig": float((img_D - image).abs().max()),
        "tie_distances": tie,
    }
except Exception:
    out["4_zero_override"] = {"error": traceback.format_exc()}

# ---------- 5: SH bwd shape variants (valid mask shapes) ----------------------
try:
    up_small = torch.randn(N, 3, device="cuda")
    up_1 = torch.randn(1, 1, N, 3, device="cuda")
    up_2 = torch.randn(1, 1, 1, N, 3, device="cuda")

    def sh_grad(means, vm, masks, up):
        s = sh_all.detach().clone().requires_grad_()
        c = (SH(degree, means, vm, s, masks=masks) + .5).clamp_min(0)
        return torch.autograd.grad(c, s, up)[0].detach()

    res = {}
    g_A = sh_grad(params["means"], cam["viewmat"][None], mask_all, up_small)          # [N,3]+[1,4,4]+[1,N]
    res["A_vs_A_nomask"] = rel(g_A, sh_grad(params["means"], cam["viewmat"][None], None, up_small))
    try:
        g_B = sh_grad(params["means"][None], cam["viewmat"][None, None], mask_all[0][None, None],
                      up_1)                                                          # [1,N,3]+[1,1,4,4]+[1,1,N]
        res["A_vs_B"] = rel(g_A, g_B)
    except Exception:
        res["A_vs_B"] = {"error": traceback.format_exc()}
    try:
        g_C = sh_grad(params["means"][None, None], cam["viewmat"][None, None, None],
                      mask_all[0][None, None, None], up_2)                           # [1,1,N,3]+[1,1,1,4,4]+[1,1,1,N]
        res["A_vs_C"] = rel(g_A, g_C)
    except Exception:
        res["A_vs_C"] = {"error": traceback.format_exc()}
    # zeroed-entry structure of A vs no-mask difference
    g_A_nm = sh_grad(params["means"], cam["viewmat"][None], None, up_small)
    dnm = (g_A - g_A_nm).abs().amax(-1)
    res["nomask_diff_rows"] = int((dnm > 0).sum())
    res["nomask_diff_rows_masked"] = int(((dnm > 0) & (~mask_all[0])).sum())
    out["5_sh_bwd_shapes"] = res
except Exception:
    out["5_sh_bwd_shapes"] = {"error": traceback.format_exc()}

# ---------- 6: conic-patch causality for the covars route ---------------------
try:
    rq, rc = meta_q["radii"][0], meta_c["radii"][0]
    valid = (rq > 0).any(-1) & (rc > 0).any(-1)
    cq = meta_q["conics"][0].clone(); cc = meta_c["conics"][0].clone()
    dab = (cq - cc).abs().amax(-1)
    drel = ((cq - cc).abs() / cq.abs().clamp_min(1e-20)).amax(-1)
    opq = meta_q["opacities"] if meta_q["opacities"].ndim == 2 else meta_q["opacities"][None]
    base_gap = float((img_c - img_q).abs().max())
    with torch.no_grad():
        img_q_re, _ = rebuild_full(meta_q["means2d"], cq[None], rq, colors, meta_q["depths"], opq)
    variants = {}
    for tag, order in (("by_abs", dab), ("by_rel", drel)):
        for kk in (16, 256, 4096, 65536):
            if kk > int(valid.sum()):
                continue
            topk = order.masked_fill(~valid, -1).topk(kk).indices
            cpatch = cc.clone(); cpatch[topk] = cq[topk]
            with torch.no_grad():
                img_p, _ = rebuild_full(meta_c["means2d"], cpatch[None], rc, colors, meta_c["depths"], opq)
            variants[f"{tag}_top{kk}"] = {
                "patched_rows": kk,
                "gap_vs_imgQ": float((img_p - img_q).abs().max()),
                "gap_vs_rebuildQ": float((img_p - img_q_re).abs().max()),
                "max_abs_conic_in_set": float(dab[topk].max()), "max_rel_conic_in_set": float(drel[topk].max()),
                "min_abs_conic_in_set": float(dab[topk].min()),
            }
    # all-row conic patch (control)
    with torch.no_grad():
        img_all, _ = rebuild_full(meta_c["means2d"], cq[None], rc, colors, meta_c["depths"], opq)
    out["6_conic_patch"] = {"base_gap_imgC_vs_imgQ": base_gap,
                            "all_conics_from_q_gap": float((img_all - img_q).abs().max()),
                            "valid_rows": int(valid.sum()),
                            "conic_abs_max": float(dab.max()), "conic_rel_max": float(drel.max()),
                            "variants": variants}
except Exception:
    out["6_conic_patch"] = {"error": traceback.format_exc()}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str)[:12000])
