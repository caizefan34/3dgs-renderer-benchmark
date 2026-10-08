"""v6b: fixed shapes and per-section saves.
(1) classify sh-grad differing entries, (2) zero-override causality restricted to
the M0 256 rows, (3) SH-bwd shape variants with valid shapes, (4) conic-patch
causality for the covars route."""
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
OUT = "/root/epic05_prism/outputs/diag_m0_v6b_bicycle_5000.json"

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


def save():
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)


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
with torch.no_grad():
    tw = math.ceil(cam["width"] / 16); th = math.ceil(cam["height"] / 16)
    _, isect_q, flat_q = isect_tiles(meta_q["means2d"], meta_q["radii"], meta_q["depths"], 16, tw, th, packed=False)
in_isect = torch.zeros(N, dtype=torch.bool, device="cuda")
in_isect[flat_q.flatten().long()] = True
print("setup done", flush=True)

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
    print("internal grads done", flush=True)
    g_r = grads(rgb_route)
    print("rgb grads done", flush=True)
    gi_all = torch.cat((g_i[4], g_i[5]), 1)   # [N,K,3]
    gr_all = torch.cat((g_r[4], g_r[5]), 1)
    d = (gi_all - gr_all).abs()
    m_leaf = mask_all[0]                                   # [N]
    ent = d > 0
    row_has = ent.flatten(1).any(-1)                       # [N]
    ent_vis = ent & m_leaf[:, None, None]
    ent_masked = ent & (~m_leaf)[:, None, None]
    iz = (gi_all == 0)
    dz = (gr_all == 0)
    out["1_entry_classes"] = {
        "n_diff_entries": int(ent.sum()),
        "n_rows_any_diff": int(row_has.sum()),
        "diff_entries_on_masked_rows": int(ent_masked.sum()),
        "diff_entries_on_visible_rows": int(ent_vis.sum()),
        "visible_rows_total": int(m_leaf.sum()),
        "masked_rows_total": int((~m_leaf).sum()),
        "diff_rows_in_isect": int((row_has & in_isect).sum()),
        "diff_rows_not_in_isect": int((row_has & ~in_isect).sum()),
        "diff_entries_channel_color_zero": int((ent & (c_draw1 == 0.0)[:, None, :]).sum()),
        "diff_entries_channel_color_lt1e-6": int((ent & (c_draw1.abs() < 1e-6)[:, None, :]).sum()),
        "rows_all_colors_zero": int((c_draw1 == 0.0).all(-1).sum()),
        "rows_near_zero_color": int((c_draw1.abs() < 1e-6).any(-1).sum()),
        "total_entries": int(ent.numel()),
        "rgb_grad_norm": float(gr_all.double().norm()), "int_grad_norm": float(gi_all.double().norm()),
        "diff_entries_internal_zero": int((ent & iz).sum()),
        "diff_entries_draw_zero": int((ent & dz).sum()),
        "internal_zero_entries_not_in_isect_rows": int((ent & iz & (~in_isect)[:, None, None]).sum()),
        "internal_zero_entries_in_isect_rows": int((ent & iz & in_isect[:, None, None]).sum()),
        "internal_zero_entries_masked_rows": int((ent & iz & (~m_leaf)[:, None, None]).sum()),
        "internal_zero_entries_at_zero_color": int((ent & iz & (c_draw1 == 0.0)[:, None, :]).sum()),
    }
    save()
    di = gi_all[ent].double(); dr = gr_all[ent].double()
    out["1_entry_classes"]["diff_mean_abs_internal_over_draw"] = float((di.abs() / dr.abs().clamp_min(1e-30)).mean())
    idx = gr_all.abs().flatten().topk(8).indices
    ex = []
    for f in idx.tolist():
        r = f // (K * 3); k = (f % (K * 3)) // 3; ch = f % 3
        ex.append({"flat": f, "row": int(r), "k": int(k), "ch": int(ch),
                   "internal": float(gi_all[r, k, ch]), "draw": float(gr_all[r, k, ch]),
                   "color_draw": float(c_draw1[r, ch]), "visible": bool(m_leaf[r]),
                   "in_isect": bool(in_isect[r]),
                   "radii": meta_q["radii"][0, r].tolist(),
                   "opacity": float(meta_q["opacities"].reshape(-1)[r])})
    out["2_top8_entries"] = ex
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
    save()
    del g_i, g_r, gi_all, gr_all, d, row_dmax
except Exception:
    out["1_entry_classes"] = {"error": traceback.format_exc()}
    save()

# ---------- 4: zero-override causality on the M0 256 rows ---------------------
try:
    image, st = B.state(params, cam, degree)
    meta = st["meta"]
    colors = st["colors"]
    radii = meta["radii"][0]
    rows_ok = (radii > 0).any(-1)
    sel = rows_ok.nonzero().flatten()[:256]
    sel_mask = torch.zeros(N, dtype=torch.bool, device="cuda"); sel_mask[sel] = True
    # roundtrip only for the selected rows (as adapter draw() does for ids)
    cov_sel = conic_to_cov(meta["conics"][0][sel])
    z_sel = screen_pack(meta["means2d"][0][sel], cov_sel)
    uv_o, cov_o = screen_unpack(z_sel)
    opac = meta["opacities"]
    if opac.ndim == 2:
        opac = opac[0]
    extend = (2 * torch.log((opac[sel] * 255).clamp_min(1))).sqrt().clamp_max(3.33)
    rr = torch.ceil(extend[:, None] * cov_o.diagonal(dim1=-2, dim2=-1).sqrt()).to(radii.dtype)
    # exact reconstruction expected for the unmodified rows
    uv_rt_max = float((uv_o - meta["means2d"][0][sel]).abs().max())
    cov_rt_max = float((cov_o - cov_sel).abs().max())
    conic_re = cov_to_conic(cov_o)
    conic_rt_max = float((conic_re - meta["conics"][0][sel]).abs().max())
    flip_sel = (rr != radii[sel]).any(-1)
    opac_t = meta["opacities"] if meta["opacities"].ndim == 2 else meta["opacities"][None]
    img_base, _ = rebuild_full(meta["means2d"], meta["conics"], radii, colors, meta["depths"], opac_t)
    # A: full override (like M0 zero=B.draw(st,valid,B.features(st,valid)))
    uv_a = meta["means2d"].clone(); co_a = meta["conics"].clone(); rr_a = radii.clone()
    uv_a[0, sel] = uv_o; co_a[0, sel] = conic_re; rr_a[sel] = rr
    img_A, _ = rebuild_full(uv_a, co_a, rr_a, colors, meta["depths"], opac_t)
    # B: override uv+conic but keep meta radii
    img_B, _ = rebuild_full(uv_a, co_a, radii, colors, meta["depths"], opac_t)
    # C: only uv replaced
    img_C, _ = rebuild_full(uv_a, meta["conics"], radii, colors, meta["depths"], opac_t)
    # D: only conic replaced
    img_D, _ = rebuild_full(meta["means2d"], co_a, radii, colors, meta["depths"], opac_t)
    tie = []
    for i in flip_sel.nonzero().flatten()[:12].tolist():
        p_a = (extend[i].double() * cov_sel[i].diagonal().double().sqrt())
        p_b = (extend[i].double() * cov_o[i].diagonal().double().sqrt())
        tie.append({"row256": int(i), "sel_row": int(sel[i]),
                    "p_orig": [float(x) for x in p_a.tolist()],
                    "p_roundtrip": [float(x) for x in p_b.tolist()],
                    "dist_to_int_p_orig": [float(min(abs(x - math.floor(x)), abs(x - math.ceil(x)))) for x in p_a.tolist()],
                    "dist_to_int_p_rt": [float(min(abs(x - math.floor(x)), abs(x - math.ceil(x)))) for x in p_b.tolist()],
                    "radii_meta": radii[sel[i]].tolist(), "radii_rt": rr[i].tolist()})
    out["4_zero_override"] = {
        "sel_rows": int(len(sel)),
        "uv_roundtrip_maxabs": uv_rt_max, "cov_roundtrip_maxabs": cov_rt_max,
        "conic_roundtrip_maxabs": conic_rt_max,
        "flip_rows_in_sel256": int(flip_sel.sum()),
        "flip_values": {str(int(v)): int(c) for v, c in
                        zip(*np.unique((rr - radii[sel])[flip_sel].cpu().numpy(), return_counts=True))},
        "sanity_base_vs_orig": float((img_base - image).abs().max()),
        "A_full_override_vs_orig": float((img_A - image).abs().max()),
        "B_radii_kept_vs_orig": float((img_B - image).abs().max()),
        "C_only_uv_vs_orig": float((img_C - image).abs().max()),
        "D_only_conic_vs_orig": float((img_D - image).abs().max()),
        "tie_distances": tie,
    }
    save()
except Exception:
    out["4_zero_override"] = {"error": traceback.format_exc()}
    save()

# ---------- 5: SH bwd shape variants (valid shapes) ---------------------------
try:
    up_0 = torch.randn(N, 3, device="cuda")          # unused if output is [1,N,3]
    up_1 = torch.randn(1, N, 3, device="cuda")
    up_2 = torch.randn(1, 1, N, 3, device="cuda")
    up_3 = torch.randn(1, 1, 1, N, 3, device="cuda")

    def sh_grad(means, vm, masks, up):
        s = sh_all.detach().clone().requires_grad_()
        c = (SH(degree, means, vm, s, masks=masks) + .5).clamp_min(0)
        if c.shape != up.shape:
            up = up.reshape(c.shape)
        g = torch.autograd.grad(c, s, up)[0].detach()
        return g, tuple(c.shape)

    g_A, sa = sh_grad(params["means"], cam["viewmat"][None], mask_all, up_1)   # [N,3]+[1,4,4]+[1,N]
    res = {"A_out_shape": list(sa)}
    g_An, sa_n = sh_grad(params["means"], cam["viewmat"][None], None, up_1)
    res["A_vs_A_nomask"] = rel(g_A, g_An)
    res["nomask_diff_rows"] = int(((g_A - g_An).abs().amax(-1) > 0).sum())
    res["nomask_diff_rows_masked"] = int((((g_A - g_An).abs().amax(-1) > 0) & (~mask_all[0])).sum())
    try:
        g_B, sb = sh_grad(params["means"][None], cam["viewmat"][None, None], mask_all[0][None, None], up_2)
        res["B_out_shape"] = list(sb)
        res["A_vs_B"] = rel(g_A, g_B)
    except Exception:
        res["A_vs_B"] = {"error": traceback.format_exc()}
    try:
        g_C, sc = sh_grad(params["means"][None, None], cam["viewmat"][None, None, None],
                          mask_all[0][None, None, None], up_3)
        res["C_out_shape"] = list(sc)
        res["A_vs_C"] = rel(g_A, g_C)
    except Exception:
        res["A_vs_C"] = {"error": traceback.format_exc()}
    out["5_sh_bwd_shapes"] = res
    save()
except Exception:
    out["5_sh_bwd_shapes"] = {"error": traceback.format_exc()}
    save()

# ---------- 6: conic-patch causality for the covars route ---------------------
try:
    colors = B.state(params, cam, degree)[1]["colors"]
    rq, rc = meta_q["radii"][0], meta_c["radii"][0]
    valid = (rq > 0).any(-1) & (rc > 0).any(-1)
    cq = meta_q["conics"][0].clone(); cc = meta_c["conics"][0].clone()
    dab = (cq - cc).abs().amax(-1)
    drel = ((cq - cc).abs() / cq.abs().clamp_min(1e-20)).amax(-1)
    opq = meta_q["opacities"] if meta_q["opacities"].ndim == 2 else meta_q["opacities"][None]
    base_gap = float((img_c - img_q).abs().max())
    with torch.no_grad():
        img_q_re, _ = rebuild_full(meta_q["means2d"], cq, rq, colors, meta_q["depths"], opq)
    variants = {}
    for tag, order in (("by_abs", dab), ("by_rel", drel)):
        for kk in (16, 256, 4096, 65536):
            if kk > int(valid.sum()):
                continue
            topk = order.masked_fill(~valid, -1).topk(kk).indices
            cpatch = cc.clone(); cpatch[topk] = cq[topk]
            with torch.no_grad():
                img_p, _ = rebuild_full(meta_c["means2d"], cpatch, rc, colors, meta_c["depths"], opq)
            variants[f"{tag}_top{kk}"] = {
                "patched_rows": kk,
                "gap_vs_imgQ": float((img_p - img_q).abs().max()),
                "max_abs_conic_in_set": float(dab[topk].max()), "min_abs_conic_in_set": float(dab[topk].min()),
            }
    with torch.no_grad():
        img_all, _ = rebuild_full(meta_c["means2d"], cq, rc, colors, meta_c["depths"], opq)
    out["6_conic_patch"] = {"base_gap_imgC_vs_imgQ": base_gap,
                            "all_conics_from_q_gap": float((img_all - img_q).abs().max()),
                            "valid_rows": int(valid.sum()),
                            "conic_abs_max": float(dab.max()), "conic_rel_max": float(drel.max()),
                            "variants": variants}
    save()
except Exception:
    out["6_conic_patch"] = {"error": traceback.format_exc()}
    save()

print(json.dumps(out, indent=2, default=str)[:12000])
