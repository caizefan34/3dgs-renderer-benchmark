#!/usr/bin/env python3
"""P2-1C-P0: Exact Forward-State-Reuse Backward Preflight measurement.

Reuses the h5-0 capture methodology (authoritative B2 forward + CPU structural
macro-hierarchy projection) and adds the P2-1C-specific analyses:

  S3  dead-group counts at 32-G mini-batch x fine-tile (recompute, must match h5-0)
  S4  weighted Gaussian-entry dead work (count_in sums, not just group counts)
  S5  last_id overlap: are dead-group entries already skipped in PROCESSING by
      current C0 V3 last_id loop?  =>  incremental removable work is the LOADING
  S6  suffix-vs-holes structure per fine tile: last_active_batch, trailing vs
      internal dead groups, fraction of tiles where dead groups are a pure suffix
  S7  metadata footprint estimates (bit-per-group, last_active frontier, active list)

ORACLE ONLY. No production kernel modified. No timing claims.
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time, uuid
from pathlib import Path
import numpy as np
import torch

TILE_SIZE = 16
SH_DEGREE = 3
K_SH = (SH_DEGREE + 1) ** 2  # 16
MTW, MTH = 8, 4
BATCH_1024 = 1024
BATCH_32 = 32
ALPHA_THRESHOLD = 1.0 / 255.0

SCENE_CONFIGS = {
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
        "native_w": 3114, "native_h": 2075,
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
        "native_w": 4946, "native_h": 3286,
    },
    "garden": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json",
        "native_w": 5187, "native_h": 3361,
    },
}


def bootstrap(source, core_so):
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core


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
    cams = json.loads(Path(cams_path).read_text())
    c = cams[0]
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


def make_fixture_and_render(scene, max_long_side, device):
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native
    from gsplat.rendering import _maybe_evaluate_sh

    cfg = SCENE_CONFIGS[scene]
    s = min(1.0, max_long_side / max(cfg["native_w"], cfg["native_h"]))
    width, height = round(cfg["native_w"] * s), round(cfg["native_h"] * s)
    tw, th = math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE)

    means, quats, scales, opacities, colors = load_ply_scene(cfg["ply"], device)
    vm, K = load_cameras(cfg["cams"], width, height, device)
    vm, K = vm[:, [0]], K[:, [0]]

    with torch.no_grad():
        ids, _, _ = _cull_gaussians_batched(
            means, quats, scales, vm, K, width, height,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            camera_model="pinhole")
        m, q, s_c, o, c = _gather_visible_native(means, quats, scales, opacities, colors, ids)
        radii, m2d, depths, conics, _ = fully_fused_projection(
            means=m.unsqueeze(0), covars=None, quats=q.unsqueeze(0), scales=s_c.unsqueeze(0),
            viewmats=vm, Ks=K, width=width, height=height,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            packed=False, calc_compensations=False, camera_model="pinhole")
        opa = o[None, None].expand(1, 1, -1).contiguous()
        try:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(),
                conics.contiguous(), opa.contiguous(), None, None, 1,
                TILE_SIZE, tw, th, True, False, None)
        except RuntimeError:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(),
                conics.contiguous(), opa.contiguous(), None, None, 1,
                TILE_SIZE, tw, th, True, False)
        offs = isect_offset_encode(isect, 1, tw, th).reshape(1, 1, th, tw)

        means_b = m[None]
        radii_b = radii
        colors_eval = _maybe_evaluate_sh(
            SH_DEGREE, c, means_b, radii_b, vm,
            (1,), 1, len(o), True).contiguous()

        m2d_b = m2d[0, 0][None, None].contiguous()
        conics_b = conics[0, 0][None, None].contiguous()
        opa_b = o[None, None].contiguous()
        offs_b = offs.contiguous()
        flat_b = flat.contiguous()
        bg = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)

        for _ in range(3):
            torch.ops.gsplat.rasterize_to_pixels_3dgs(
                m2d_b, conics_b, colors_eval, opa_b, bg, None,
                width, height, TILE_SIZE, offs_b, flat_b, False, False)
        torch.cuda.synchronize()

        result = torch.ops.gsplat.rasterize_to_pixels_3dgs(
            m2d_b, conics_b, colors_eval, opa_b, bg, None,
            width, height, TILE_SIZE, offs_b, flat_b, False, False)
        if len(result) == 4:
            render_colors, render_alphas, _absgrad, last_ids = result
        elif len(result) == 3:
            render_colors, render_alphas, last_ids = result
        else:
            raise RuntimeError(f"Unexpected number of return values: {len(result)}")

    state = {
        "scene": scene, "width": width, "height": height,
        "tw": tw, "th": th,
        "m2d": m2d[0, 0].cpu().numpy(),
        "conics": conics[0, 0].cpu().numpy(),
        "depth": depths[0, 0].cpu().numpy(),
        "opacity": o.cpu().numpy(),
        "radii": radii[0, 0].cpu().numpy().astype(np.int32),
        "flat": flat.cpu().numpy().astype(np.int32),
        "offs": offs[0, 0].cpu().numpy(),
        "n_vis": len(o),
        "last_ids": last_ids[0, 0].cpu().numpy().astype(np.int32),
    }
    return state


def compute_macro_representation(state):
    tw, th = state["tw"], state["th"]
    mw, mh = math.ceil(tw / MTW), math.ceil(th / MTH)
    m2d = state["m2d"]
    conics = state["conics"]
    depth = state["depth"]
    opacity = state["opacity"]
    MAX_EXTEND = 4096.0

    def emit(A, B, C, t, cx, cy, cols, rows, sx, sy):
        disc = B * B - A * C
        if not (disc < 0 and t > 0):
            return []
        ex = math.sqrt(-t * C / disc)
        ey = math.sqrt(-t * A / disc)
        xmin, xmax = cx - ex, cx + ex
        ymin, ymax = cy - ey, cy + ey
        rx0 = max(0, min(cols, int(xmin / sx)))
        rx1 = max(0, min(cols, int(xmax / sx + 1)))
        ry0 = max(0, min(rows, int(ymin / sy)))
        ry1 = max(0, min(rows, int(ymax / sy + 1)))
        out = []
        for y in range(ry0, ry1):
            for x in range(rx0, rx1):
                x0, x1 = x * sx, (x + 1) * sx
                y0, y1 = y * sy, (y + 1) * sy
                pts = [(min(max(cx, x0), x1), min(max(cy, y0), y1))]
                for xx in (x0, x1):
                    yy = min(max(cy - B * (xx - cx) / C, y0), y1)
                    pts.append((xx, yy))
                for yy in (y0, y1):
                    xx = min(max(cx - B * (yy - cy) / A, x0), x1)
                    pts.append((xx, yy))
                q = min(A * (xx - cx) ** 2 + 2 * B * (xx - cx) * (yy - cy) + C * (yy - cy) ** 2 for xx, yy in pts)
                if q <= t:
                    out.append(y * cols + x)
        return out

    macros = [[] for _ in range(mw * mh)]
    fine = [[] for _ in range(tw * th)]
    for g in range(len(m2d)):
        cx, cy = float(m2d[g, 0]), float(m2d[g, 1])
        A, B, C = float(conics[g, 0]), float(conics[g, 1]), float(conics[g, 2])
        o = float(opacity[g])
        z = float(depth[g])
        if not (o >= ALPHA_THRESHOLD and A > 0 and C > 0):
            continue
        t = min(MAX_EXTEND * MAX_EXTEND, 2 * math.log(o / ALPHA_THRESHOLD))
        mts = emit(A, B, C, t, cx, cy, mw, mh, MTW * TILE_SIZE, MTH * TILE_SIZE)
        tiles = emit(A, B, C, t, cx, cy, tw, th, TILE_SIZE, TILE_SIZE)
        for mt in mts:
            macros[mt].append(g)
        for tile in tiles:
            fine[tile].append(g)

    for xs in macros:
        xs.sort(key=lambda g: (float(depth[g]), g))
    for xs in fine:
        xs.sort(key=lambda g: (float(depth[g]), g))

    return {
        "macros": macros,
        "fine": fine,
        "mw": mw, "mh": mh,
        "n_macro_entries": sum(len(xs) for xs in macros),
        "n_fine_pairs": sum(len(xs) for xs in fine),
    }


def group_analysis(state, macro_rep, batch_size=32):
    """Per-(mini-batch, fine-tile) active/dead with weighted entries and structure."""
    tw, th = state["tw"], state["th"]
    mw, mh = macro_rep["mw"], macro_rep["mh"]
    offs = state["offs"]
    flat_len = len(state["flat"])
    last_ids = state["last_ids"]
    H, W = state["height"], state["width"]

    off_flat = np.append(offs.ravel(), flat_len)
    macros = macro_rep["macros"]
    fine = macro_rep["fine"]

    # L_tile = max last_ids over pixels in tile (absolute)
    L_tile = np.full(tw * th, -1, dtype=np.int64)
    for py in range(H):
        for px in range(W):
            tid = (py // TILE_SIZE) * tw + (px // TILE_SIZE)
            lid = int(last_ids[py, px])
            if lid > L_tile[tid]:
                L_tile[tid] = lid
    L_tile_local = np.where(L_tile >= 0, L_tile - off_flat[:-1], -1)

    total_groups = 0
    dead_groups = 0
    active_groups = 0
    dead_by_nocover = 0          # dead because no entry in batch covers tile
    dead_by_lastid = 0           # dead because count_before > L_tile_local
    dead_weight_entries = 0      # sum over dead groups of count_in
    total_weight_entries = 0     # sum over ALL groups of count_in (== n_fine_pairs)
    active_weight_entries = 0

    # suffix-vs-holes per fine tile
    tile_last_active = {}        # ft_id -> last active batch index (within macro)
    tile_trailing_dead = {}      # ft_id -> number of dead groups after last_active
    tile_internal_dead = {}      # ft_id -> number of dead groups before last_active
    tile_has_entries = {}        # ft_id -> True if tile has any covering entry

    for mt in range(mw * mh):
        entries = macros[mt]
        n_entries = len(entries)
        if n_entries == 0:
            continue
        base = (mt // mw) * MTH * tw + (mt % mw) * MTW

        fine_sets = [set(fine[base + (j // MTW) * tw + (j % MTW)]) if (base + (j // MTW) * tw + (j % MTW)) < tw * th else set()
                     for j in range(MTW * MTH)]
        entry_masks = []
        for g in entries:
            mask = 0
            for j in range(MTW * MTH):
                ft_id = base + (j // MTW) * tw + (j % MTW)
                if ft_id < tw * th and g in fine_sets[j]:
                    mask |= (1 << j)
            entry_masks.append(mask)
        entry_masks = np.array(entry_masks, dtype=np.uint32)

        n_batches = (n_entries + batch_size - 1) // batch_size
        covers_all = entry_masks  # [n_entries, 1] uint32 bit per fine tile j

        for j in range(MTW * MTH):
            ft_id = base + (j // MTW) * tw + (j % MTW)
            if ft_id < 0 or ft_id >= tw * th:
                continue
            bit = np.uint32(1 << j)
            covers = ((covers_all & bit) != 0).astype(np.int32)
            cumcount = np.cumsum(covers) - covers  # count_before per entry

            lt = int(L_tile_local[ft_id])
            per_batch_active = np.zeros(n_batches, dtype=bool)
            per_batch_weight = np.zeros(n_batches, dtype=np.int64)   # count_in per batch
            per_batch_nocover = np.zeros(n_batches, dtype=bool)
            per_batch_lastid = np.zeros(n_batches, dtype=bool)

            for b in range(n_batches):
                bs = b * batch_size
                be = min(bs + batch_size, n_entries)
                cnt_before = int(cumcount[bs]) if be > bs else 0
                cnt_in = int(np.sum(covers[bs:be]))
                total_weight_entries += cnt_in
                total_groups += 1
                if cnt_in == 0:
                    dead_groups += 1
                    dead_by_nocover += 1
                    per_batch_nocover[b] = True
                    continue
                per_batch_weight[b] = cnt_in
                if lt < 0 or cnt_before > lt:
                    dead_groups += 1
                    dead_by_lastid += 1
                    per_batch_lastid[b] = True
                else:
                    active_groups += 1
                    per_batch_active[b] = True
                    active_weight_entries += cnt_in

            # weighted dead entries
            for b in range(n_batches):
                if not per_batch_active[b]:
                    dead_weight_entries += int(per_batch_weight[b])

            # suffix / holes per tile
            if n_batches == 0:
                continue
            last_a = -1
            for b in range(n_batches - 1, -1, -1):
                if per_batch_active[b]:
                    last_a = b
                    break
            tile_has_entries[ft_id] = bool(np.any(per_batch_weight > 0))
            if last_a < 0:
                tile_last_active[ft_id] = -1
                tile_trailing_dead[ft_id] = n_batches
                tile_internal_dead[ft_id] = 0
            else:
                tile_last_active[ft_id] = last_a
                trailing = int(np.sum(~per_batch_active[last_a + 1:]))
                internal = int(np.sum(~per_batch_active[:last_a]))
                tile_trailing_dead[ft_id] = trailing
                tile_internal_dead[ft_id] = internal

    # stats
    def stats(arr):
        vals = np.asarray(arr, dtype=np.float64)
        if len(vals) == 0:
            return {"mean": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "n": 0}
        return {
            "mean": float(np.mean(vals)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(np.max(vals)),
            "n": int(len(vals)),
        }

    last_active_vals = [v for v in tile_last_active.values()]
    trailing_vals = [v for v in tile_trailing_dead.values()]
    internal_vals = [v for v in tile_internal_dead.values()]
    tiles_with_entries = [ft for ft, h in tile_has_entries.items() if h]

    pure_suffix_tiles = sum(1 for ft in tiles_with_entries if tile_internal_dead.get(ft, 0) == 0)

    return {
        "scene": state["scene"],
        "batch_size": batch_size,
        "total_groups": total_groups,
        "active_groups": active_groups,
        "dead_groups": dead_groups,
        "dead_fraction": dead_groups / total_groups if total_groups else 0,
        "dead_by_nocover_groups": dead_by_nocover,
        "dead_by_lastid_groups": dead_by_lastid,
        "total_weight_entries": total_weight_entries,
        "active_weight_entries": active_weight_entries,
        "dead_weight_entries": dead_weight_entries,
        "dead_weight_fraction": dead_weight_entries / total_weight_entries if total_weight_entries else 0,
        "last_active_batch_per_tile": stats(last_active_vals),
        "trailing_dead_per_tile": stats(trailing_vals),
        "internal_dead_per_tile": stats(internal_vals),
        "n_tiles_with_entries": len(tiles_with_entries),
        "n_tiles_pure_suffix": pure_suffix_tiles,
        "pure_suffix_fraction": pure_suffix_tiles / len(tiles_with_entries) if tiles_with_entries else 0,
    }


def last_id_overlap(state, macro_rep, ga, batch_size=32):
    """P2-1C S5: overlap between hierarchy-dead work and current last_id skip.

    Current C0 V3 backward (per h5-0 last_id_semantics.json):
      - warp-level loop: t starts at max(0, batch_end - warp_bin_final); if
        batch_end - warp_bin_final >= batch_size, the batch's loop body does
        not run at all -> whole batch skipped in PROCESSING.
      - per-pixel: valid=0 for lanes beyond bin_final.
      - DATA LOADING (lines 198-212) is UNCONDITIONAL for all batches.

    A dead (mini-batch x fine-tile) group has all its entries beyond
    L_tile_local (tile-max last_id), hence beyond every pixel's bin_final and
    beyond warp_bin_final for every warp inside the tile -> 100% of the
    group's PROCESSING is already removed by last_id. The incremental
    P2-1C opportunity is avoiding the LOAD for those groups (and their
    control/path setup), which the current backward still performs.
    """
    tw, th = state["tw"], state["th"]
    mw, mh = macro_rep["mw"], macro_rep["mh"]
    offs = state["offs"]
    flat_len = len(state["flat"])
    last_ids = state["last_ids"]
    H, W = state["height"], state["width"]

    off_flat = np.append(offs.ravel(), flat_len)
    macros = macro_rep["macros"]
    fine = macro_rep["fine"]

    L_tile = np.full(tw * th, -1, dtype=np.int64)
    for py in range(H):
        for px in range(W):
            tid = (py // TILE_SIZE) * tw + (px // TILE_SIZE)
            lid = int(last_ids[py, px])
            if lid > L_tile[tid]:
                L_tile[tid] = lid
    L_tile_local = np.where(L_tile >= 0, L_tile - off_flat[:-1], -1)

    # h5-0 "already skipped processing entries" per-tile:
    # entries with tile-local rank > L_tile_local are never processed.
    tile_list_lengths = np.diff(off_flat)
    already_skipped_entries = 0
    total_entries = 0
    dead_group_entries = 0      # entries inside dead groups (== weighted dead work)
    dead_group_already_skipped = 0  # of those, already skipped by last_id (processing)
    for tid in range(tw * th):
        tlen = int(tile_list_lengths[tid])
        total_entries += tlen
        if tlen > 0 and L_tile_local[tid] >= 0:
            already_skipped_entries += max(0, tlen - int(L_tile_local[tid]) - 1)

    # For every dead group, its entries lie beyond L_tile_local by construction,
    # so dead_group_already_skipped == dead_group_entries (processing already removed).
    # Recompute the dead-group entry count independently here for cross-check.
    for mt in range(mw * mh):
        entries = macros[mt]
        n_entries = len(entries)
        if n_entries == 0:
            continue
        base = (mt // mw) * MTH * tw + (mt % mw) * MTW
        fine_sets = [set(fine[base + (j // MTW) * tw + (j % MTW)]) if (base + (j // MTW) * tw + (j % MTW)) < tw * th else set()
                     for j in range(MTW * MTH)]
        entry_masks = []
        for g in entries:
            mask = 0
            for j in range(MTW * MTH):
                ft_id = base + (j // MTW) * tw + (j % MTW)
                if ft_id < tw * th and g in fine_sets[j]:
                    mask |= (1 << j)
            entry_masks.append(mask)
        entry_masks = np.array(entry_masks, dtype=np.uint32)
        n_batches = (n_entries + batch_size - 1) // batch_size
        for j in range(MTW * MTH):
            ft_id = base + (j // MTW) * tw + (j % MTW)
            if ft_id < 0 or ft_id >= tw * th:
                continue
            bit = np.uint32(1 << j)
            covers = ((entry_masks & bit) != 0).astype(np.int32)
            cumcount = np.cumsum(covers) - covers
            lt = int(L_tile_local[ft_id])
            for b in range(n_batches):
                bs = b * batch_size
                be = min(bs + batch_size, n_entries)
                cnt_in = int(np.sum(covers[bs:be]))
                if cnt_in == 0:
                    continue
                cnt_before = int(cumcount[bs]) if be > bs else 0
                if lt >= 0 and cnt_before <= lt:
                    continue  # active group
                dead_group_entries += cnt_in
                # all cnt_in entries have rank >= cnt_before > lt == L_tile_local
                # -> beyond every pixel's last_id -> processing already skipped
                dead_group_already_skipped += cnt_in

    return {
        "scene": state["scene"],
        "gross_hierarchy_dead_work_entries": dead_group_entries,
        "already_removed_by_last_id_processing": dead_group_already_skipped,
        "last_id_overlap_fraction": (dead_group_already_skipped / dead_group_entries
                                     if dead_group_entries else 0),
        "already_skipped_processing_entries_total": already_skipped_entries,
        "total_entries": total_entries,
        "already_skipped_processing_fraction": already_skipped_entries / total_entries if total_entries else 0,
        "incremental_removable_load_entries": dead_group_entries,
        "incremental_note": "dead groups are a SUBSET of the last_id-skipped tail; their processing is 100% already removed. Incremental P2-1C value = avoiding the UNCONDITIONAL DATA LOAD (sorted IDs + attributes + control) for these groups. No double count of processing work.",
    }


def batch256_analysis(state, macro_rep):
    """C0 V3 compatible analysis: the frozen baseline (higs_blend_bwd_kernel)
    iterates each FINE tile's flatten_ids range in batches of block_size=256
    gaussians, loading all 256 (id/means2d/opacity/conics/colors) into smem
    UNCONDITIONALLY. A batch is fully-dead (skippable before load) iff its
    entire index range is beyond the tile's max last_id (L_tile_local).

    Returns the fully-dead 256-G batch counts + entries, plus the 32-G
    fully-dead counts for granularity comparison.
    """
    tw, th = state["tw"], state["th"]
    offs = state["offs"]
    flat_len = len(state["flat"])
    last_ids = state["last_ids"]
    H, W = state["height"], state["width"]
    off_flat = np.append(offs.ravel(), flat_len)

    L_tile = np.full(tw * th, -1, dtype=np.int64)
    for py in range(H):
        for px in range(W):
            tid = (py // TILE_SIZE) * tw + (px // TILE_SIZE)
            lid = int(last_ids[py, px])
            if lid > L_tile[tid]:
                L_tile[tid] = lid
    L_tile_local = np.where(L_tile >= 0, L_tile - off_flat[:-1], -1)

    BS = 128  # C0 V3 deployed config: higs_blend_bwd_px_kernel<CDIM=3,PX=2> block_size=128
    total_batches = 0
    dead_batches = 0
    dead_entries = 0
    total_entries = 0
    tail_entries = 0
    for tid in range(tw * th):
        tlen = int(off_flat[tid + 1] - off_flat[tid])
        total_entries += tlen
        if tlen == 0 or L_tile_local[tid] < 0:
            total_batches += (tlen + BS - 1) // BS
            continue
        lt = int(L_tile_local[tid])
        num_batches = (tlen + BS - 1) // BS
        total_batches += num_batches
        tail = max(0, tlen - 1 - lt)  # entries beyond every pixel's last_id
        tail_entries += tail
        # batches from back: batch b covers [batch_end-255, batch_end]
        # fully-dead iff batch_end - 255 > lt  <=>  first index > lt
        b = 0
        while b < num_batches:
            batch_end = tlen - 1 - BS * b
            first_idx = batch_end - (BS - 1)
            if first_idx > lt:
                dead_batches += 1
                dead_entries += batch_end + 1 - max(first_idx, 0)
                b += 1
            else:
                break  # deeper batches contain the frontier -> active

    return {
        "scene": state["scene"],
        "block_size": BS,
        "total_batches": total_batches,
        "fully_dead_batches": dead_batches,
        "fully_dead_batch_fraction": dead_batches / total_batches if total_batches else 0,
        "fully_dead_batch_entries": dead_entries,
        "total_entries": total_entries,
        "fully_dead_batch_entry_fraction": dead_entries / total_entries if total_entries else 0,
        "tail_entries_beyond_lastid": tail_entries,
        "note": "C0 V3 PX=2 block_size=128: a 128-G batch is load-skippable only if ALL 128 indices exceed the tile's max last_id. 32-G forward masks map 4:1 onto these batches; the block-uniform predicate (L_tile_local is tile-wide) gives a divergence-free early continue.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="/tmp/higs_h3_fwd_1a_source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--max-long-side", type=int, default=2048)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PATH"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    run_id = str(uuid.uuid4())[:12]
    print(f"=== P2-1C-P0 measurement run_id={run_id} GPU={args.gpu} ===", flush=True)

    bootstrap(args.source, args.core_so)
    print("Bootstrap complete.", flush=True)

    all_group = {}
    all_overlap = {}
    all_b256 = {}
    for scene in ["room", "bicycle", "garden"]:
        print(f"\n=== {scene} ===", flush=True)
        state = make_fixture_and_render(scene, args.max_long_side, device)
        print(f"  {state['width']}x{state['height']}, N_vis={state['n_vis']}, n_isects={len(state['flat'])}", flush=True)
        macro_rep = compute_macro_representation(state)
        print(f"  macro entries={macro_rep['n_macro_entries']}, fine pairs={macro_rep['n_fine_pairs']}", flush=True)
        ga = group_analysis(state, macro_rep, BATCH_32)
        all_group[scene] = ga
        print(f"  dead_fraction={ga['dead_fraction']:.4f} dead_weight={ga['dead_weight_fraction']:.4f} "
              f"pure_suffix={ga['pure_suffix_fraction']:.4f}", flush=True)
        lo = last_id_overlap(state, macro_rep, ga, BATCH_32)
        all_overlap[scene] = lo
        print(f"  lastid_overlap={lo['last_id_overlap_fraction']:.4f} "
              f"incremental_load={lo['incremental_removable_load_entries']}", flush=True)
        b256 = batch256_analysis(state, macro_rep)
        all_b256[scene] = b256
        print(f"  b256: dead_batches={b256['fully_dead_batch_fraction']:.4f} "
              f"dead_entries={b256['fully_dead_batch_entry_fraction']:.4f}", flush=True)

    (out_dir / "dead_group_counts.json").write_text(json.dumps(
        {s: {k: all_group[s][k] for k in
             ["scene", "batch_size", "total_groups", "active_groups", "dead_groups",
              "dead_fraction", "dead_by_nocover_groups", "dead_by_lastid_groups"]}
         for s in all_group}, indent=2))
    (out_dir / "weighted_dead_work.json").write_text(json.dumps(
        {s: {k: all_group[s][k] for k in
             ["scene", "total_weight_entries", "active_weight_entries",
              "dead_weight_entries", "dead_weight_fraction"]}
         for s in all_group}, indent=2))
    (out_dir / "suffix_vs_holes.json").write_text(json.dumps(
        {s: {k: all_group[s][k] for k in
             ["scene", "last_active_batch_per_tile", "trailing_dead_per_tile",
              "internal_dead_per_tile", "n_tiles_with_entries", "n_tiles_pure_suffix",
              "pure_suffix_fraction"]}
         for s in all_group}, indent=2))
    (out_dir / "last_id_overlap.json").write_text(json.dumps(
        {s: {k: all_overlap[s][k] for k in
             ["scene", "gross_hierarchy_dead_work_entries",
              "already_removed_by_last_id_processing", "last_id_overlap_fraction",
              "already_skipped_processing_entries_total", "total_entries",
              "already_skipped_processing_fraction", "incremental_removable_load_entries",
              "incremental_note"]}
         for s in all_overlap}, indent=2))
    (out_dir / "batch256_work.json").write_text(json.dumps(
        {s: {k: all_b256[s][k] for k in
             ["scene", "block_size", "total_batches", "fully_dead_batches",
              "fully_dead_batch_fraction", "fully_dead_batch_entries", "total_entries",
              "fully_dead_batch_entry_fraction", "tail_entries_beyond_lastid", "note"]}
         for s in all_b256}, indent=2))

    core_so_sha = hashlib.sha256(Path(args.core_so).read_bytes()).hexdigest() if Path(args.core_so).exists() else "N/A"
    provenance = {
        "run_id": run_id,
        "task": "P2-1C-P0",
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "gpu": f"cuda:{args.gpu}",
        "source": args.source,
        "core_so": args.core_so,
        "core_so_sha256": core_so_sha,
        "max_long_side": args.max_long_side,
        "method": "h5-0 capture methodology (authoritative B2 forward + CPU macro projection) + P2-1C weighted / last_id-overlap / suffix-vs-holes extensions",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
