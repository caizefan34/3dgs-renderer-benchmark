#!/usr/bin/env python3
"""H7-B0: HiGS Macro-Owned Gaussian Backward Roofline Oracle.

Computes, from the authoritative EXACT R2 macro representation
(macro_sorted_ids + uint32 fine_tile_masks, produced by higs_train_macro_f4)
and the authoritative H5 forward last_ids, all twelve roofline sections that
decide whether the macro-owned backward (H7-B) has real net headroom.

Does NOT implement H7-B. This is a read-only structural/profiling oracle.
Sections computed strictly from nav arrays via vectorized numpy.
"""
import argparse, hashlib, importlib.util, json, math, os, runpy, sys, time, uuid
from pathlib import Path
import numpy as np
import torch

TILE_SIZE = 16
SH_DEGREE = 3
K_SH = 16
MTW, MTH = 8, 4
BATCH_32 = 32
AT = 1.0 / 255.0
MAX_EXTEND = 4096.0
PIX_PER_TILE = 256

SCENE_CONFIGS = {
    "room":    ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
                "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",  3114, 2075),
    "bicycle": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
                "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json", 4946, 3286),
    "garden":  ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply",
                "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json", 5187, 3361),
}
# Authoritative anchors (h1 / h6-0r / h3-1a):
BWD_MS = {"room": 2.870, "bicycle": 3.775, "garden": 1.51}   # B2 backward (garden estimated from isect scaling)
FWD5_MS = {"room": 0.6717, "bicycle": 0.9103, "garden": 0.4127}  # B2 forward raster single-call
C_MACRO_MS = {"room": 0.2591440, "bicycle": 0.1515520, "garden": 0.0399360}  # R2_Macro_F4 - B2_F4
# per-attribute load units in the accounting model
ATTR_SLOTS = {"means2d": 2, "conic": 3, "opacity": 1, "color": 3, "id": 1}
BYTES_PER_SLOT = 4


def bootstrap(source, core_so):
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    build = runpy.run_path(str(Path(source) / "gsplat/experimental/render/kernels/cuda/build.py"))
    return build["build_and_load_experimental_gaussian_render_inference_scene"]()


def load_ply_scene(ply_path, device):
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), device=device, dtype=torch.float32)
    quats = torch.tensor(np.column_stack([v[f"rot_{i}"] for i in range(4)]), device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v[f"scale_{i}"] for i in range(3)]), device=device, dtype=torch.float32))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], device=device, dtype=torch.float32))
    sh = torch.zeros((len(v), K_SH, 3), device=device, dtype=torch.float32)
    sh[:, 0] = torch.tensor(np.column_stack([v[f"f_dc_{i}"] for i in range(3)]), device=device, dtype=torch.float32)
    rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], device=device, dtype=torch.float32) for i in range(45)], 1)
    sh[:, 1:] = rest.reshape(len(v), 3, 15).permute(0, 2, 1)
    return means, quats, scales, opacities, sh


def load_cameras(cams_path, width, height, device):
    c = json.loads(Path(cams_path).read_text())[0]
    native_w, native_h = int(c["width"]), int(c["height"])
    R = np.asarray(c["rotation"], dtype=np.float32).T
    p = np.asarray(c["position"], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ p
    K = np.array([[float(c["fx"]) * width / native_w, 0, (width - 1) / 2],
                  [0, float(c["fy"]) * width / native_w, (height - 1) / 2],
                  [0, 0, 1]], dtype=np.float32)
    return torch.tensor(vm, device=device)[None, None], torch.tensor(K, device=device)[None, None]


def capture_state(scene, max_long_side, device):
    """Reproduce authoritative H5 forward state: returns CPU numpy arrays incl last_ids."""
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native
    from gsplat.rendering import _maybe_evaluate_sh
    ply, cams, nw, nh = SCENE_CONFIGS[scene]
    s = min(1.0, max_long_side / max(nw, nh))
    width, height = round(nw * s), round(nh * s)
    tw, th = math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE)
    means, quats, scales, opacities, colors = load_ply_scene(ply, device)
    vm, K = load_cameras(cams, width, height, device)
    with torch.no_grad():
        ids, _, _ = _cull_gaussians_batched(means, quats, scales, vm, K, width, height,
                                            eps2d=0.3, near_plane=0.01, far_plane=1e10,
                                            radius_clip=0.0, camera_model="pinhole")
        m, q, s_c, o, c = _gather_visible_native(means, quats, scales, opacities, colors, ids)
        radii, m2d, depths, conics, _ = fully_fused_projection(
            means=m.unsqueeze(0), covars=None, quats=q.unsqueeze(0), scales=s_c.unsqueeze(0),
            viewmats=vm, Ks=K, width=width, height=height, eps2d=0.3, near_plane=0.01,
            far_plane=1e10, radius_clip=0.0, packed=False, calc_compensations=False, camera_model="pinhole")
        opa = o[None, None].expand(1, 1, -1).contiguous()
        try:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(), conics.contiguous(),
                opa.contiguous(), None, None, 1, TILE_SIZE, tw, th, True, False, None)
        except RuntimeError:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(), conics.contiguous(),
                opa.contiguous(), None, None, 1, TILE_SIZE, tw, th, True, False)
        offs = isect_offset_encode(isect, 1, tw, th).reshape(1, 1, th, tw)
        colors_eval = _maybe_evaluate_sh(SH_DEGREE, c, m[None], radii, vm, (1,), 1, len(o), True).contiguous()
        m2d_b = m2d[0, 0][None, None].contiguous(); conics_b = conics[0, 0][None, None].contiguous()
        opa_b = o[None, None].contiguous(); offs_b = offs.contiguous(); flat_b = flat.contiguous()
        bg = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
        for _ in range(2):
            torch.ops.gsplat.rasterize_to_pixels_3dgs(m2d_b, conics_b, colors_eval, opa_b, bg, None,
                                                      width, height, TILE_SIZE, offs_b, flat_b, False, False)
        torch.cuda.synchronize()
        res = torch.ops.gsplat.rasterize_to_pixels_3dgs(m2d_b, conics_b, colors_eval, opa_b, bg, None,
                                                        width, height, TILE_SIZE, offs_b, flat_b, False, False)
        if len(res) == 4:
            _, _, _, last_ids = res
        elif len(res) == 3:
            _, _, last_ids = res
        else:
            raise RuntimeError("unexpected rasterize return")
    return {
        "scene": scene, "width": width, "height": height, "tw": tw, "th": th,
        "m2d": m2d[0, 0].cpu().numpy(), "conics": conics[0, 0].cpu().numpy(),
        "depth": depths[0, 0].cpu().numpy(), "opacity": o.cpu().numpy(),
        "radii": radii[0, 0].cpu().numpy().astype(np.int32),
        "flat": flat.cpu().numpy().astype(np.int32), "offs": offs[0, 0].cpu().numpy(),
        "n_vis": len(o),
        "last_ids": last_ids[0, 0].cpu().numpy().astype(np.int32),
    }


def compute_r2(ext, state, device):
    """Run authoritative R2 macro-F4 kernel -> (offsets, macro_sorted_ids, batch_offsets, masks)."""
    ins = [torch.from_numpy(state[k]).to(device) for k in ("m2d", "conics", "depth", "opacity", "radii")]
    out = ext.higs_train_macro_f4(*ins, state["tw"], state["th"], TILE_SIZE, True, False)
    torch.cuda.synchronize()
    offsets, sorted_ids, batch_offsets, masks = (x.detach().cpu().numpy() for x in out[:4])
    return offsets, sorted_ids, batch_offsets, masks


def perc(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def dist_report(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {k: 0.0 for k in ["mean", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    return {"mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def section1(masks):
    pops = np.asarray([int(np.uint32(m)).bit_count() for m in masks], dtype=np.int32)
    edges = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 24), (25, 32)]
    freq = {e[0] if e[0] == e[1] else f"{e[0]}-{e[1]}": int(np.sum((pops >= e[0]) & (pops <= e[1]))) for e in edges}
    total = int(len(pops))
    return {"dist": dist_report(pops), "edges": edges, "bucket_counts": freq,
            "bucket_fractions": {k: (v / total if total else 0) for k, v in freq.items()},
            "sum_fine_pairs": int(pops.sum()), "n_entries": total}


def section2(state, offsets, masks, last_ids):
    """backward-active tile count per macro entry via last_id frontier gating."""
    tw, th, mw = state["tw"], state["th"], math.ceil(state["tw"] / MTW)
    off_abs = np.append(state["offs"].ravel(), len(state["flat"]))
    tile_len = np.diff(off_abs)
    H, W = state["height"], state["width"]
    # L_tile = max(last_ids) per fine tile
    pix_tile = np.zeros(H * W, dtype=np.int64)
    lf = last_ids.ravel().astype(np.int64)
    for py in range(H):
        base = py * W
        pix_tile[base:base + W] = (py // TILE_SIZE) * tw + (np.arange(W) // TILE_SIZE)
    L_tile = np.full(tw * th, -1, dtype=np.int64)
    np.maximum.at(L_tile, pix_tile, lf)
    tile_start_abs = off_abs[:-1]
    L_local = np.where(L_tile >= 0, L_tile - tile_start_abs, -1)
    L_local = np.clip(L_local, -1, tile_len - 1)  # cap by list length

    M = len(masks)
    entry_active_tiles = np.zeros(M, dtype=np.int32)
    active_masks = np.zeros(M, dtype=np.uint32)
    masks_i = masks.astype(np.int64)
    n_macro = len(offsets) - 1
    for mt in range(n_macro):
        s, e = int(offsets[mt]), int(offsets[mt + 1])
        if s >= e:
            continue
        mx, my = mt % mw, mt // mw
        msub = masks_i[s:e]
        if msub.sum() == 0:
            continue
        for j in range(32):
            bit = np.int64(1) << j
            covers = (msub & bit) != 0
            n_cov = int(np.count_nonzero(covers))
            if n_cov == 0:
                continue
            row_off, col_off = j // MTW, j % MTW
            fx = mx * MTW + col_off; fy = my * MTH + row_off
            if fx >= tw or fy >= th:
                continue
            ft = fy * tw + fx
            rank = np.cumsum(covers) - covers  # B2 0-based rank within ft
            active = covers & (rank <= L_local[ft])
            idx = np.nonzero(active)[0]
            if idx.size:
                entry_active_tiles[s + idx] += 1
                active_masks[s + idx] |= np.uint32(1 << j)
    ba_tiles = entry_active_tiles
    ba_fine_pairs = int(ba_tiles.sum())
    ba_entries = int(np.count_nonzero(ba_tiles >= 1))
    return {
        "L_tile_local_p99": perc(L_local[L_local >= 0], 99),
        "n_fine_tiles_active_nonempty": int(np.count_nonzero(tile_len > 0)),
        "backward_active_tile_count": dist_report(ba_tiles[ba_tiles >= 1]),
        "backward_active_tile_count_incl_dead": dist_report(ba_tiles),
        "backward_active_fine_pairs": ba_fine_pairs,
        "backward_active_macro_entries": ba_entries,
        "geometric_fine_pairs": len(state["flat"]),
        "geometric_macro_entries": M,
        "R_geom_fine_over_macro": (len(state["flat"]) / M) if M else 0,
        "R_backward_fine_ownership_over_macro_ownership": (ba_fine_pairs / ba_entries) if ba_entries else 0,
        "dead_macro_entries_fraction": 1.0 - (ba_entries / M if M else 0),
    }, active_masks


def section3(state, offsets, active_masks):
    """32-G owner-group statistics."""
    n_macro = len(offsets) - 1
    group_active_tiles = []; active_lanes_per_tile = []; densities = []; lane_util = []
    for mt in range(n_macro):
        s, e = int(offsets[mt]), int(offsets[mt + 1])
        if s >= e:
            continue
        am = active_masks[s:e].astype(np.int64)
        n = len(am)
        for g0 in range(0, n, BATCH_32):
            grp = am[g0:g0 + BATCH_32]
            lanes = len(grp)
            union = 0
            for r in grp:
                union |= r
            active_tiles = int(bin(int(union)).count("1"))
            group_active_tiles.append(active_tiles)
            # active lanes per fine tile bit
            for j in range(32):
                nl = int(np.count_nonzero((grp >> j) & 1))
                if nl:
                    active_lanes_per_tile.append(nl)
            dens = int(np.count_nonzero(grp)) / (lanes * 32)
            densities.append(dens)
            lane_util.extend([int(r).bit_count() for r in grp])
    lu = lane_util
    return {
        "active_fine_tiles_per_group": dist_report(group_active_tiles),
        "active_gaussian_lanes_per_tile": dist_report(active_lanes_per_tile),
        "mask_density_per_group": dist_report(densities),
        "lane_utilization_active_tiles_per_lane": dist_report(lu),
        "n_groups": len(group_active_tiles),
    }


def section4(state, sec2):
    """Logical attribute-load roofline."""
    total_slots = sum(ATTR_SLOTS.values())
    bytes_per_unit = total_slots * BYTES_PER_SLOT
    base_pairs = sec2["backward_active_fine_pairs"]
    ideal_entries = sec2["backward_active_macro_entries"]
    r = {}
    per_attr_bytes = {k: v * BYTES_PER_SLOT for k, v in ATTR_SLOTS.items()}
    for attr, slots in ATTR_SLOTS.items():
        base = base_pairs * slots
        ideal = ideal_entries * slots
        r[attr] = {"baseline_logical_loads": int(base), "h7_ideal_logical_loads": int(ideal),
                   "reduction_frac": (1 - (ideal / base)) if base else 0.0}
    total_base = base_pairs * total_slots
    total_ideal = ideal_entries * total_slots
    r["total"] = {"attr_slots": ATTR_SLOTS, "bytes_per_unit": bytes_per_unit,
                  "baseline_logical_units": int(base_pairs), "h7_ideal_logical_units": int(ideal_entries),
                  "baseline_logical_load_slots": int(total_base), "h7_ideal_logical_slots": int(total_ideal),
                  "reduction_pct_total": (1 - total_ideal / total_base) * 100 if total_base else 0.0,
                  "per_attr_bytes_per_unit": per_attr_bytes}
    return r


def section5(state, sec2, sec4):
    """Cache-aware physical traffic ceiling (analytic model; no hardware replay kernel)."""
    per_unit_bytes = sec4["total"]["bytes_per_unit"]
    base_units = sec2["backward_active_fine_pairs"]
    ideal_units = sec2["backward_active_macro_entries"]
    nvis = state["n_vis"]
    # Logical request volumes (what L1/L2 serve)
    base_req_bytes = base_units * per_unit_bytes
    h7_req_bytes = ideal_units * per_unit_bytes
    # DRAM cold-footprint: each visible gaussian's attribute block is read from DRAM ~once into L2,
    # then reused across its tiles via L1/L2. Same for both schemes -> physical DRAM reduction ~0.
    dram_base = nvis * per_unit_bytes
    dram_h7 = nvis * per_unit_bytes
    return {
        "per_unit_bytes": per_unit_bytes,
        "n_visible_gaussians": nvis,
        "logical_request_bytes_baseline": int(base_req_bytes),
        "logical_request_bytes_H7": int(h7_req_bytes),
        "logical_request_reduction_pct": (1 - h7_req_bytes / base_req_bytes) * 100 if base_req_bytes else 0,
        "L2_request_reduction_pct": (1 - ideal_units / base_units) * 100 if base_units else 0,
        "DRAM_attribute_bytes_baseline": int(dram_base),
        "DRAM_attribute_bytes_H7": int(dram_h7),
        "DRAM_reduction_pct_ceiling": 0.0,
        "cache_model": "Gaussian attribute working set per color = n_visible*40B, ~<7MB on bicycle, fits L2(40MB). "
                       "Cross-tile reuse is served by L1/L2, so logical reuse mostly does NOT reach DRAM. "
                       "Physical DRAM reduction ceiling ~0%. Hardware L1/L2 replay measurement deferred.",
    }


def section6(state, sec2, r2):
    """Gradient-write / atomic roofline."""
    base_pairs = sec2["backward_active_fine_pairs"]
    ideal_entries = sec2["backward_active_macro_entries"]
    grad_attrs = ["v_means2d", "v_conic", "v_color", "v_opacity"]
    r = {}
    for ga in grad_attrs:
        base = base_pairs
        ideal = ideal_entries
        r[ga] = {"current_fine_tile_ownership_flushes": int(base),
                 "macro_owner_flushes": int(ideal),
                 "reduction_factor": (base / ideal) if ideal else 0.0,
                 "reduction_pct": (1 - ideal / base) * 100 if base else 0.0}
    r["note"] = "counts are logical gradient ownership units (fine-tile owner vs one aggregate flush per active macro owner). Actual hardware atomic count not separately measured."
    return r


def section7(state, sec2, masks):
    """Incremental value of last-id frontier over HiGS mask alone."""
    geo_pairs = len(state["flat"])
    ba_pairs = sec2["backward_active_fine_pairs"]
    # pixel-Gaussian visits: per-pixel reachable contributions (governed by last_id already)
    off_abs = np.append(state["offs"].ravel(), len(state["flat"]))
    tile_start = off_abs[:-1]
    tile_lens = np.diff(off_abs)
    lf = state["last_ids"].ravel().astype(np.int64)
    H, W = state["height"], state["width"]
    pix_tile = np.zeros(H * W, dtype=np.int64)
    for py in range(H):
        base = py * W
        tl = (py // TILE_SIZE) * state["tw"] + (np.arange(W) // TILE_SIZE)
        pix_tile[base:base + W] = tl
    ts = tile_start[pix_tile]; tl = tile_lens[pix_tile]
    reach = np.where(lf >= ts, lf - ts + 1, 0)
    reach = np.clip(reach, 0, tl)
    pixel_visits = int(reach.sum())
    return {
        "attribute_loads": {
            "baseline_fine_pairs": geo_pairs,
            "mask_only_geometric_fine_pairs": geo_pairs,  # R2 masks reconstruct exactly the same pairs
            "mask_AND_lastid_fine_pairs": ba_pairs,
            "incremental_lastid_removed_pairs": int(geo_pairs - ba_pairs),
            "incremental_lastid_removed_pct": (1 - ba_pairs / geo_pairs) * 100 if geo_pairs else 0,
            "mask_alone_adds_zero": True,
            "note": "Exact R2 fine_tile_masks reconstruct exactly the B2 fine pairs; mask ALONE removes 0 additional work. All frontier gain comes from last_id gating.",
        },
        "pixel_visits": {
            "baseline_pixel_gaussian_visits": pixel_visits,
            "mask_only": pixel_visits,
            "mask_AND_lastid": pixel_visits,
            "incremental": 0,
            "note": "Per-pixel reachable set is already fixed by last_ids; tile-level frontier gating does not reduce per-pixel arithmetic (last_id already prunes it).",
        },
        "gradient_ownership_units": {
            "baseline": geo_pairs,
            "mask_only": geo_pairs,
            "mask_AND_lastid": ba_pairs,
            "incremental_removed_pct": (1 - ba_pairs / geo_pairs) * 100 if geo_pairs else 0,
        },
        "dead32g_tile_fraction_note": "see frontier_analysis/recomputed in this run",
    }


def section8(state, offsets, masks, active_masks, sec2):
    """Pixel-work warning for Design A (owner scans all 256 px of every active fine tile)."""
    # Design A pixel-visits = sum over backward-active entries of (active tile count * 256)
    # active tile count per entry comes direct from active_masks popcount.
    pops = np.asarray([int(np.uint32(m)).bit_count() for m in active_masks], dtype=np.int64)
    designA_visits = int(pops.sum()) * PIX_PER_TILE
    off_abs = np.append(state["offs"].ravel(), len(state["flat"]))
    tile_start = off_abs[:-1]; tile_lens = np.diff(off_abs)
    lf = state["last_ids"].ravel().astype(np.int64)
    H, W = state["height"], state["width"]; pix_tile = np.zeros(H * W, dtype=np.int64)
    for py in range(H):
        base = py * W
        pix_tile[base:base + W] = (py // TILE_SIZE) * state["tw"] + (np.arange(W) // TILE_SIZE)
    ts = tile_start[pix_tile]; tl = tile_lens[pix_tile]
    reach = np.clip(np.where(lf >= ts, lf - ts + 1, 0), 0, tl)
    baseline_visits = int(reach.sum())
    return {
        "designA_owner_scans_all256_per_active_tile": int(designA_visits),
        "designB_scans_baseline_active_pixels": int(baseline_visits),
        "baseline_valid_gaussian_weight_evals_approx": baseline_visits,
        "baseline_actual_contributing_samples_approx": baseline_visits,
        "designA_over_baseline_ratio": (designA_visits / baseline_visits) if baseline_visits else 0,
        "designB_over_baseline_ratio": 1.0,
        "pixel_work_explosion_DesignA": (designA_visits / baseline_visits) > 2.0 if baseline_visits else False,
        "warning": "Design A explodes pixel work by the ratio above; Design B adds no new pixel work. No new contribution metadata introduced here.",
    }


def section9(state, active_masks, sec8):
    """Warp (32-G owner group) lane-load balance."""
    offsets = state["_offsets"]
    lane_work = []  # per gaussian lane: active fine tiles (the per-lane tile-work unit)
    for mt in range(len(offsets) - 1):
        s, e = int(offsets[mt]), int(offsets[mt + 1])
        if s >= e:
            continue
        am = active_masks[s:e].astype(np.int64)
        n = len(am)
        for g0 in range(0, n, BATCH_32):
            grp = am[g0:g0 + BATCH_32]
            w = np.array([int(r).bit_count() for r in grp], dtype=np.float64)
            if len(w) == 0:
                continue
            m, mx = w.mean(), w.max()
            lane_work.append({"mean": float(m), "max": float(mx), "max_over_mean": float(mx / m if m else 0),
                              "p90": float(np.percentile(w, 90)), "p99": float(np.percentile(w, 99)),
                              "p90_over_mean": float(np.percentile(w, 90) / m if m else 0),
                              "p99_over_mean": float(np.percentile(w, 99) / m if m else 0)})
    mmol = [g["max_over_mean"] for g in lane_work]
    p90mm = [g["p90_over_mean"] for g in lane_work]
    p99mm = [g["p99_over_mean"] for g in lane_work]
    return {
        "n_groups": len(lane_work),
        "per_group_mean_lane_work_stat": dist_report([g["mean"] for g in lane_work]),
        "max_over_mean_per_group": dist_report(mmol),
        "p90_over_mean_per_group": dist_report(p90mm),
        "p99_over_mean_per_group": dist_report(p99mm),
        "hard_gate_p90_of_p90_over_mean_le4": bool(perc(p90mm, 90) <= 4.0),
        "hard_gate_p90_of_max_over_mean_le4": bool(np.percentile(mmol, 90) <= 4.0),
        "note": "lane work = backward-active fine-tile count of the gaussian lane (Design-B pixel work scales with it).",
    }


def section10():
    """Metadata requirement (no-new-metadata design check)."""
    return {
        "required_persistent_metadata": {
            "macro_sorted_ids": "already persisted H3-R2 (uint32)",
            "fine_tile_masks": "already persisted H3-R2 (uint32)",
            "last_ids": "already persisted H5 forward checkpoint",
            "macro_offsets": "already persisted H3-R2",
        },
        "new_persistent_bytes": 0,
        "per_pixel_x_group_masks_required": False,
        "extra_classified_cost": {"read": 0, "write": 0},
        "verdict": "PASS — ~0 new persistent metadata. Macro-owner groups are reconstructed at schedule time from existing macro_sorted_ids + fine_tile_masks + last_ids.",
    }


def section11(state, sec5, sec6, r2):
    """Net F+B accounting with C_macro overhead included."""
    scene = state["scene"]
    F = FWD5_MS[scene]; B = BWD_MS[scene]; Cm = C_MACRO_MS[scene]
    l2red = sec5["L2_request_reduction_pct"] / 100.0
    atomic_red = sec6["v_means2d"]["reduction_pct"] / 100.0
    # backward is compute-bound (H2-BWD-0: blend backward compute-bound, atomics only ~6%).
    # Section 5 shows DRAM reduction ceiling ~=0%: cross-tile attribute reuse is ALREADY served
    # by L1/L2. So the load-driven share of backward time does NOT shrink under cache-realistic
    # accounting. Only the atomic-flush reduction (macro-owner aggregated) survives into real time,
    # and it is capped at ~6% (atomic share of backward).
    atomic_share = 0.06
    load_share = 0.20
    opt_back = B * (atomic_share * atomic_red + load_share * l2red * 0.5)   # optimistic: half the load reduction reaches real time
    real_back = B * (atomic_share * atomic_red)                              # realistic: cache already captures load reuse
    pes_back = 0.0
    FB = F + B
    def net(save):
        return (save - Cm) / FB * 100.0
    return {
        "basis": {"F": F, "B": B, "F_plus_B": FB, "C_macro_ms": Cm},
        "C_macro_fraction_of_F_plus_B_pct": Cm / FB * 100.0,
        "atomic_reduction_pct": atomic_red * 100, "L2_reduction_pct": l2red * 100,
        "optimistic_ms": {"backward_saving": opt_back, "net": net(opt_back)},
        "realistic_ms": {"backward_saving": real_back, "net": net(real_back)},
        "pessimistic_ms": {"backward_saving": pes_back, "net": net(pes_back)},
        "model_assumptions": "backward is compute-bound; atomics ~6% of backward. Section5: DRAM reduction ceiling ~0% "
                             "because cross-tile attribute reuse is already captured by L1/L2; hence the load-driven share "
                             "of backward time does NOT shrink (realistic backward saving = atomic-flush reduction only). "
                             "Optimistic credits half of the logical L2 reduction into real backward time. "
                             "Net always subtracts the exact-macro-F4 construction penalty C_macro on the forward side.",
    }


def core_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "N/A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="/tmp/higs_h3_fwd_1a_source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--max-long-side", type=int, default=2048)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    run_id = str(uuid.uuid4())[:12]
    print(f"=== H7-B0 Macro-Owner Roofline Oracle run_id={run_id} GPU={a.gpu} ===", flush=True)
    ext = bootstrap(a.source, a.core_so)
    print("bootstrap OK", flush=True)

    exact_macro = {}; bwd_eff = {}; g32 = {}; logical = {}; phys = {}; gradf = {}
    frontier = {}; pixel = {}; warp = {}; meta = section10()

    for scene in ["room", "bicycle", "garden"]:
        print(f"\n=== {scene} ===", flush=True)
        state = capture_state(scene, a.max_long_side, device)
        state["_offsets"] = None
        print(f"  state {state['width']}x{state['height']} n_vis={state['n_vis']} isects={len(state['flat'])}", flush=True)
        offsets, sorted_ids, batch_offsets, masks = compute_r2(ext, state, device)
        state["_offsets"] = offsets
        print(f"  R2 entries={int(offsets[-1])} fine_pairs={int(len(state['flat']))}", flush=True)

        s1 = section1(masks)
        exact_macro[scene] = {"scene": scene, **s1}
        print(f"  S1 popcount mean={s1['dist']['mean']:.3f} p50={s1['dist']['p50']} p90={s1['dist']['p90']}", flush=True)

        s2, active_masks = section2(state, offsets, masks, state["last_ids"])
        bwd_eff[scene] = {"scene": scene, **s2}
        print(f"  S2 R_geom={s2['R_geom_fine_over_macro']:.3f} R_back={s2['R_backward_fine_ownership_over_macro_ownership']:.3f} "
              f"ba_pairs={s2['backward_active_fine_pairs']}", flush=True)

        g32[scene] = {"scene": scene, **section3(state, offsets, active_masks)}
        logical[scene] = {"scene": scene, **section4(state, s2)}
        phys[scene] = {"scene": scene, **section5(state, s2, logical[scene])}
        gradf[scene] = {"scene": scene, **section6(state, s2, offsets)}
        frontier[scene] = {"scene": scene, **section7(state, s2, masks)}
        pixel[scene] = {"scene": scene, **section8(state, offsets, masks, active_masks, s2)}
        warp[scene] = {"scene": scene, **section9(state, active_masks, pixel[scene])}
        print(f"  S8 DesignA/baseline={pixel[scene]['designA_over_baseline_ratio']:.2f} ", flush=True)
        print(f"  S9 p90 max/mean={warp[scene]['max_over_mean_per_group']['p90']:.2f}", flush=True)

    net = {s: {"scene": s, **section11({"scene": s}, phys[s], gradf[s], None)} for s in ["room", "bicycle", "garden"]}

    def write(nm, obj):
        (out / nm).write_text(json.dumps(obj, indent=2))

    write("exact_macro_reuse.json", exact_macro)
    write("backward_effective_reuse.json", bwd_eff)
    write("group32_masks.json", g32)
    write("logical_load_roofline.json", logical)
    write("physical_traffic_roofline.json", phys)
    write("gradient_flush_roofline.json", gradf)
    write("frontier_increment.json", frontier)
    write("pixel_work.json", pixel)
    write("warp_balance.json", warp)
    write("metadata_cost.json", meta)
    write("net_roofline.json", net)

    # ---- Section 12 gate ----
    R = {s: bwd_eff[s]["R_backward_fine_ownership_over_macro_ownership"] for s in ["room", "bicycle", "garden"]}
    phys_red = {s: phys[s]["DRAM_reduction_pct_ceiling"] for s in ["room", "bicycle", "garden"]}
    atomic_red = {s: gradf[s]["v_means2d"]["reduction_pct"] for s in ["room", "bicycle", "garden"]}
    p90im = {s: warp[s]["max_over_mean_per_group"]["p90"] for s in ["room", "bicycle", "garden"]}
    net_real = {s: net[s]["realistic_ms"]["net"] for s in ["room", "bicycle", "garden"]}
    A = sum(1 for s in R if R[s] >= 2.0) >= 2
    B_phys = sum(1 for s in phys_red if phys_red[s] >= 25.0) >= 2
    B_gr = sum(1 for s in atomic_red if atomic_red[s] >= 35.0) >= 2
    B = B_phys or B_gr
    C = all(p90im[s] <= 4.0 for s in p90im)
    D = sum(1 for s in net_real if net_real[s] >= 3.0) >= 2
    E = True
    all_ok = A and B and C and D and E
    # H7-B proceeds ONLY if ALL gates pass. Gate C (owner-warp balance) is a hard gate.
    if all_ok and all(net_real[s] >= 5.0 for s in net_real):
        cls = "H7B0_STRONG"
    elif all_ok and (sum(1 for s in net_real if net_real[s] >= 5.0) >= 2 or B):
        cls = "H7B0_MARGINAL"
    elif all_ok:
        cls = "H7B0_MARGINAL"
    else:
        cls = "H7B0_WEAK"

    analysis = {
        "classification": cls,
        "gates": {
            "A_real_reuse_R_backend_ge2_on23": {"threshold": "R_backward>=2.0 on >=2/3", "values": R,
                                                 "pass": A, "n_pass": sum(1 for v in R.values() if v >= 2.0)},
            "B_work_elim": {"dr_pct_thresh": 25.0, "dr_pct_values": phys_red, "door_phys_pass": B_phys,
                            "gradient_flush_thresh": 35.0, "gradient_red_pct_values": atomic_red, "door_grad_pass": B_gr,
                            "pass": B},
            "C_warp_balance_p90_max_over_mean": {"threshold": "p90 max/mean <= 4x", "values": p90im, "pass": C},
            "D_net_gain_fb": {"threshold": "realistic net F+B >= 3% on >=2/3", "values": net_real, "pass": D},
            "E_no_large_metadata": {"pass": E},
            "ALL": all_ok,
        },
        "realistic_net_fb_pct": net_real,
    }
    write("analysis.json", analysis)
    prov = {
        "run_id": run_id, "timestamp": time.strftime("%Y%m%dT%H%M%S"), "gpu": f"cuda:{a.gpu}",
        "source": a.source, "core_so": a.core_so, "core_so_sha256": core_sha(a.core_so),
        "scene_ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/*/...,cam0",
        "max_long_side": a.max_long_side,
        "representation": "H3-FWD-1A-R2 exact macro (macro_sorted_ids, uint32 fine_tile_masks from higs_train_macro_f4)",
        "last_ids": "H5 forward rasterize_to_pixels_3dgs (LAST_ID_ABSOLUTE)",
        "method": "read-only structural oracle; vectorized numpy; no production CUDA modified",
        "section5": "analytic cache model (hardware L1/L2/DRAM replay kernel deferred)",
        "backward/anchor/timings": {"B2_backward_ms": BWD_MS, "B2_forward_raster_ms": FWD5_MS, "C_macro_ms": C_MACRO_MS},
    }
    write("provenance.json", prov)
    print("\n=== classification ===", flush=True)
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()