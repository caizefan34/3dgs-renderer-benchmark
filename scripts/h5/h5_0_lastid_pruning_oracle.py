#!/usr/bin/env python3
"""H5-0: Last-ID Guided Hierarchical Backward Pruning Oracle.

Sections:
1. Verify last_ids semantic contract (source-verified, not inferred)
2. Capture exact real-scene state
3. Measure current per-pixel tail pruning
4. Project last_ids onto HiGS hierarchy (1024-G batch level)
5. Repeat at 32-G mini-batch level
6. Estimate exact removable work
7. Pixel-level active masks
8. (Optional) Oracle timing — skipped unless structural analysis is strong
9. Gate decision

Does NOT implement a production pruning kernel.
"""
import argparse, hashlib, importlib.util, json, math, os, runpy, sys, time, uuid
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
    """Produce F0-F4 fixture + run F5 to get last_ids. Returns CPU numpy state."""
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

        # SH-evaluated colors for F5
        # _maybe_evaluate_sh(sh_degree, colors[N,K,D], means[1,N,3],
        #   radii[1,1,N,2], viewmats[1,1,4,4], (1,), C, N, True) -> [1, C, N, 3]
        means_b = m[None]  # [1, N, 3]
        radii_b = radii  # already [1, 1, N, 2] from fully_fused_projection
        colors_input = c  # [N, K, 3]
        colors_eval = _maybe_evaluate_sh(
            SH_DEGREE, colors_input, means_b, radii_b, vm,
            (1,), 1, len(o), True).contiguous()  # [1, C, N, 3]

        # F5 call to get last_ids
        m2d_b = m2d[0, 0][None, None].contiguous()
        conics_b = conics[0, 0][None, None].contiguous()
        opa_b = o[None, None].contiguous()
        offs_b = offs.contiguous()
        flat_b = flat.contiguous()
        bg = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            torch.ops.gsplat.rasterize_to_pixels_3dgs(
                m2d_b, conics_b, colors_eval, opa_b, bg, None,
                width, height, TILE_SIZE, offs_b, flat_b, False, False)
        torch.cuda.synchronize()

        # Single call to get last_ids (we don't need timing here)
        result = torch.ops.gsplat.rasterize_to_pixels_3dgs(
            m2d_b, conics_b, colors_eval, opa_b, bg, None,
            width, height, TILE_SIZE, offs_b, flat_b, False, False)

        # Unpack — the combined op returns 4 values: (render_colors, render_alphas, absgrad, last_ids)
        if len(result) == 4:
            render_colors, render_alphas, _absgrad, last_ids = result
        elif len(result) == 3:
            render_colors, render_alphas, last_ids = result
        else:
            raise RuntimeError(f"Unexpected number of return values: {len(result)}")

    # Move to CPU numpy
    state = {
        "scene": scene, "width": width, "height": height,
        "tw": tw, "th": th,
        "m2d": m2d[0, 0].cpu().numpy(),        # [N_vis, 2]
        "conics": conics[0, 0].cpu().numpy(),   # [N_vis, 3]
        "depth": depths[0, 0].cpu().numpy(),    # [N_vis]
        "opacity": o.cpu().numpy(),              # [N_vis]
        "radii": radii[0, 0].cpu().numpy().astype(np.int32),
        "flat": flat.cpu().numpy().astype(np.int32),   # [n_isects]
        "offs": offs[0, 0].cpu().numpy(),        # [th, tw]
        "n_vis": len(o),
        "last_ids": last_ids[0, 0].cpu().numpy().astype(np.int32),  # [H, W]
        "render_alphas": render_alphas[0, 0, :, :, 0].cpu().numpy(),
    }
    return state


def compute_macro_representation(state):
    """Compute the HiGS macro tile representation on CPU (from h3_fwd_0_oracle.oracle)."""
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

    # Build macro and fine lists
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

    # Sort by (depth, gaussian_id)
    for xs in macros:
        xs.sort(key=lambda g: (float(depth[g]), g))
    for xs in fine:
        xs.sort(key=lambda g: (float(depth[g]), g))

    # Compute masks for each macro entry
    # mask bit j is set iff the Gaussian covers fine tile at position j within the macro tile
    # j = fine_tile_y_within_macro * MTW + fine_tile_x_within_macro
    # fine_tile_id = base + (j // MTW) * tw + (j % MTW)
    # base = (mt // mw) * MTH * tw + (mt % mw) * MTW
    macro_entries = []  # list of (macro_tile_id, gaussian_id, mask)
    for mt, xs in enumerate(macros):
        base = (mt // mw) * MTH * tw + (mt % mw) * MTW
        for g in xs:
            mask = 0
            for j in range(MTW * MTH):
                ft_id = base + (j // MTW) * tw + (j % MTW)
                if ft_id < tw * th and g in set(fine[ft_id]):
                    mask |= (1 << j)
            macro_entries.append((mt, g, mask))

    return {
        "macros": macros,
        "fine": fine,
        "macro_entries": macro_entries,  # [(mt, g, mask), ...]
        "mw": mw, "mh": mh,
        "n_macro_entries": len(macro_entries),
        "n_fine_pairs": sum(len(xs) for xs in fine),
    }


def section3_per_pixel_tail_stats(state):
    """Section 3: Measure current per-pixel tail pruning."""
    offs = state["offs"]  # [th, tw]
    flat_len = len(state["flat"])
    last_ids = state["last_ids"]  # [H, W]
    H, W = state["height"], state["width"]
    tw, th = state["tw"], state["th"]

    # For each pixel, compute tile_list_length and reachable_length
    # tile_list_length = tile_offsets[tile+1] - tile_offsets[tile]
    # last_id = last_ids[pixel] (absolute position in flatten_ids)
    # reachable_length = last_id - tile_offsets[tile] + 1 (tile-local rank + 1)
    # tail_skip_fraction = (tile_list_length - reachable_length) / tile_list_length

    off_flat = np.append(offs.ravel(), flat_len)
    tile_list_lengths = np.diff(off_flat)  # [tw*th]

    # For each pixel, get its tile
    tile_ids = np.zeros(H * W, dtype=np.int32)
    for py in range(H):
        for px in range(W):
            ty = py // TILE_SIZE
            tx = px // TILE_SIZE
            tile_ids[py * W + px] = ty * tw + tx

    last_ids_flat = last_ids.ravel()  # [H*W]

    # Only consider inside pixels (last_ids > 0 means a Gaussian was reached)
    # last_ids == 0 means no Gaussian contributed (or the first one at position 0)
    # Actually, cur_idx starts at 0 and is only updated when a Gaussian contributes
    # If no Gaussian contributes, last_ids = 0, which could be ambiguous
    # But for tiles with 0 entries, tile_list_length = 0, so we skip those

    has_entries = tile_list_lengths[tile_ids] > 0
    # For pixels with last_ids > 0 or in tiles with entries
    valid = has_entries.copy()

    tile_starts = off_flat[tile_ids]  # [H*W] tile start in flatten_ids
    tile_lens = tile_list_lengths[tile_ids]  # [H*W]

    # reachable_length = last_id - tile_start + 1
    # But last_id could be 0 even when tile has entries (no Gaussian reached threshold)
    # In that case reachable_length = 0
    reachable = np.where(
        valid & (last_ids_flat >= tile_starts),
        last_ids_flat - tile_starts + 1,
        0
    )
    reachable = np.clip(reachable, 0, tile_lens)

    tail_skip = np.where(
        tile_lens > 0,
        (tile_lens - reachable) / tile_lens,
        0.0
    )

    # Stats for valid pixels only
    valid_mask = valid & (tile_lens > 0)

    def stats(arr, mask):
        vals = arr[mask]
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

    return {
        "scene": state["scene"],
        "tile_list_length": stats(tile_lens.astype(np.float64), valid_mask),
        "reachable_length": stats(reachable.astype(np.float64), valid_mask),
        "tail_skip_fraction": stats(tail_skip, valid_mask),
        "n_valid_pixels": int(np.sum(valid_mask)),
        "n_total_pixels": int(H * W),
        "note": "tail_skip_fraction = (tile_list_length - reachable_length) / tile_list_length. This is the work ALREADY skipped by the current per-pixel last_id loop. It is baseline context, NOT new optimization opportunity.",
    }


def section4_batch_level(state, macro_rep, batch_size):
    """Section 4/5: Project last_ids onto HiGS hierarchy at given batch level."""
    tw, th = state["tw"], state["th"]
    mw, mh = macro_rep["mw"], macro_rep["mh"]
    offs = state["offs"]
    flat_len = len(state["flat"])
    last_ids = state["last_ids"]
    H, W = state["height"], state["width"]

    off_flat = np.append(offs.ravel(), flat_len)
    macros = macro_rep["macros"]
    fine = macro_rep["fine"]
    macro_entries = macro_rep["macro_entries"]

    # For each fine tile, compute L_tile = max(last_ids[pixel]) for all pixels in tile
    L_tile = np.full(tw * th, -1, dtype=np.int64)
    for py in range(H):
        for px in range(W):
            ty = py // TILE_SIZE
            tx = px // TILE_SIZE
            tid = ty * tw + tx
            lid = int(last_ids[py, px])
            if lid > L_tile[tid]:
                L_tile[tid] = lid

    # L_tile_local = L_tile - tile_offsets[tile] (tile-local rank of last reachable)
    L_tile_local = np.where(
        L_tile >= 0,
        L_tile - off_flat[:-1],
        -1
    )

    # For each macro tile, process its entries in batches
    total_batches = 0
    fully_dead_batches = 0
    partially_active_batches = 0
    fully_active_batches = 0

    total_batch_tile_groups = 0
    dead_batch_tile_groups = 0
    active_batch_tile_groups = 0

    # For active_tile_mask statistics
    active_mask_popcounts = []

    # Per-batch dead fraction
    batch_dead_fractions = []

    for mt in range(mw * mh):
        entries = macros[mt]
        n_entries = len(entries)
        if n_entries == 0:
            continue

        base = (mt // mw) * MTH * tw + (mt % mw) * MTW

        # Compute masks for each entry in this macro tile
        entry_masks = []
        fine_sets = [set(fine[base + (j // MTW) * tw + (j % MTW)]) if (base + (j // MTW) * tw + (j % MTW)) < tw * th else set()
                     for j in range(MTW * MTH)]
        for g in entries:
            mask = 0
            for j in range(MTW * MTH):
                ft_id = base + (j // MTW) * tw + (j % MTW)
                if ft_id < tw * th and g in fine_sets[j]:
                    mask |= (1 << j)
            entry_masks.append(mask)

        entry_masks = np.array(entry_masks, dtype=np.uint32)

        # Number of batches
        n_batches = (n_entries + batch_size - 1) // batch_size

        # For each fine tile within this macro tile, compute the cumulative count
        # of entries covering it (to determine tile-local ranks)
        # fine_tile_ft_in_macro: j -> ft_id
        ft_ids_in_macro = []
        for j in range(MTW * MTH):
            ft_id = base + (j // MTW) * tw + (j % MTW)
            if ft_id < tw * th:
                ft_ids_in_macro.append((j, ft_id))
            else:
                ft_ids_in_macro.append((j, -1))

        # Precompute: for each fine tile j in macro, cumulative count of entries covering it
        # cumcount_before[i] = number of entries at positions < i that cover fine tile j
        for j, ft_id in ft_ids_in_macro:
            if ft_id < 0:
                continue
            # Bit mask for this fine tile
            bit = np.uint32(1 << j)
            covers = (entry_masks & bit) != 0
            cumcount = np.cumsum(covers.astype(np.int32)) - covers.astype(np.int32)
            # cumcount[i] = number of entries before position i that cover ft_id
            # This gives the B2 tile-local rank of entry i (if it covers ft_id)

        # Now process batches
        for b in range(n_batches):
            batch_start = b * batch_size
            batch_end = min(batch_start + batch_size, n_entries)
            batch_masks = entry_masks[batch_start:batch_end]

            # active_tile_mask: bit j is set iff at least one pixel in fine tile j
            # could reach at least one Gaussian in this batch
            active_tile_mask = np.uint32(0)
            batch_dead_tiles = 0
            batch_active_tiles = 0
            total_tiles_in_macro = 0

            for j, ft_id in ft_ids_in_macro:
                if ft_id < 0:
                    continue
                total_tiles_in_macro += 1
                bit = np.uint32(1 << j)

                # Does any entry in this batch cover ft_id?
                covers_in_batch = (batch_masks & bit) != 0
                if not np.any(covers_in_batch):
                    # No entry in batch covers this fine tile
                    batch_dead_tiles += 1
                    total_batch_tile_groups += 1
                    dead_batch_tile_groups += 1
                    continue

                # Count entries before this batch that cover ft_id
                covers_all = (entry_masks & bit) != 0
                count_before = int(np.sum(covers_all[:batch_start]))

                # Count entries in this batch that cover ft_id
                count_in = int(np.sum(covers_in_batch))

                # B2 tile-local ranks for entries in this batch: count_before, ..., count_before+count_in-1
                # The batch is dead for this tile if count_before > L_tile_local[ft_id]
                lt = int(L_tile_local[ft_id])
                if lt < 0:
                    # No pixel in this tile has any last_id (empty tile or all pixels outside)
                    batch_dead_tiles += 1
                    total_batch_tile_groups += 1
                    dead_batch_tile_groups += 1
                elif count_before > lt:
                    # All entries in this batch are after the last reachable one
                    batch_dead_tiles += 1
                    total_batch_tile_groups += 1
                    dead_batch_tile_groups += 1
                else:
                    # At least one entry is reachable
                    active_tile_mask |= bit
                    batch_active_tiles += 1
                    total_batch_tile_groups += 1
                    active_batch_tile_groups += 1

            total_batches += 1
            if batch_active_tiles == 0:
                fully_dead_batches += 1
            elif batch_dead_tiles == 0 and total_tiles_in_macro > 0:
                fully_active_batches += 1
            else:
                partially_active_batches += 1

            if total_tiles_in_macro > 0:
                batch_dead_fractions.append(batch_dead_tiles / total_tiles_in_macro)

            if batch_active_tiles > 0:
                active_mask_popcounts.append(int(bin(active_tile_mask).count('1')))

    return {
        "batch_size": batch_size,
        "total_batches": total_batches,
        "fully_dead_batches": fully_dead_batches,
        "partially_active_batches": partially_active_batches,
        "fully_active_batches": fully_active_batches,
        "batch_dead_fraction": fully_dead_batches / total_batches if total_batches > 0 else 0,
        "total_batch_tile_groups": total_batch_tile_groups,
        "dead_batch_tile_groups": dead_batch_tile_groups,
        "active_batch_tile_groups": active_batch_tile_groups,
        "batch_tile_dead_fraction": dead_batch_tile_groups / total_batch_tile_groups if total_batch_tile_groups > 0 else 0,
        "active_tile_popcount_per_batch": {
            "mean": float(np.mean(active_mask_popcounts)) if active_mask_popcounts else 0,
            "p50": float(np.percentile(active_mask_popcounts, 50)) if active_mask_popcounts else 0,
            "p90": float(np.percentile(active_mask_popcounts, 90)) if active_mask_popcounts else 0,
            "p95": float(np.percentile(active_mask_popcounts, 95)) if active_mask_popcounts else 0,
            "p99": float(np.percentile(active_mask_popcounts, 99)) if active_mask_popcounts else 0,
        },
        "per_batch_dead_tile_fraction": {
            "mean": float(np.mean(batch_dead_fractions)) if batch_dead_fractions else 0,
            "p50": float(np.percentile(batch_dead_fractions, 50)) if batch_dead_fractions else 0,
            "p90": float(np.percentile(batch_dead_fractions, 90)) if batch_dead_fractions else 0,
            "p95": float(np.percentile(batch_dead_fractions, 95)) if batch_dead_fractions else 0,
            "p99": float(np.percentile(batch_dead_fractions, 99)) if batch_dead_fractions else 0,
        },
    }


def section6_removable_work(state, macro_rep, batch_stats_1024, batch_stats_32):
    """Section 6: Estimate exact load/traversal savings."""
    tw, th = state["tw"], state["th"]
    offs = state["offs"]
    flat_len = len(state["flat"])
    last_ids = state["last_ids"]
    H, W = state["height"], state["width"]

    off_flat = np.append(offs.ravel(), flat_len)
    tile_list_lengths = np.diff(off_flat)

    # ALREADY_REMOVED_BY_CURRENT_LAST_ID:
    # The current backward already skips PROCESSING (but not data loading) for
    # entries after bin_final per pixel, and uses warp_bin_final for warp-level skip.
    # Estimate: for each pixel, entries after last_id are skipped in processing.
    # But data loading still happens for all batches.

    # Compute per-tile already-skipped entries
    L_tile = np.full(tw * th, -1, dtype=np.int64)
    for py in range(H):
        for px in range(W):
            ty = py // TILE_SIZE
            tx = px // TILE_SIZE
            tid = ty * tw + tx
            lid = int(last_ids[py, px])
            if lid > L_tile[tid]:
                L_tile[tid] = lid

    already_skipped_entries = 0
    total_entries = 0
    for tid in range(tw * th):
        tlen = int(tile_list_lengths[tid])
        total_entries += tlen
        if tlen > 0 and L_tile[tid] >= 0:
            lt_local = L_tile[tid] - int(off_flat[tid])
            already_skipped_entries += max(0, tlen - lt_local - 1)

    # NEW_HIERARCHICAL_REMOVABLE_WORK:
    # At 1024-G batch level: dead batch×tile groups mean we can skip loading
    # sorted IDs, masks, and Gaussian attributes for those groups.
    # At 32-G mini-batch level: dead mini-batch×tile groups mean even finer skip.

    new_dead_1024 = batch_stats_1024["dead_batch_tile_groups"]
    new_dead_32 = batch_stats_32["dead_batch_tile_groups"]

    # Estimate removable loads:
    # Each dead batch×tile group means we skip loading:
    # - 1024 (or 32) sorted IDs from global memory
    # - 1024 (or 32) mask loads (if macro representation)
    # - For active entries: Gaussian attribute loads (means2d, conics, colors, opacities)
    # But the backward loads data cooperatively (each thread loads 1 Gaussian per batch),
    # so a dead batch×tile group means the entire batch's data loading can be skipped
    # for that tile. However, the backward uses block-level cooperative loading,
    # so the skip would need to be at the block level (all tiles in the block).

    # More precisely: the NEW opportunity is to skip entire batches (block-level)
    # where ALL tiles in the block are dead. This is the fully_dead_batches count.
    fully_dead_1024 = batch_stats_1024["fully_dead_batches"]
    fully_dead_32 = batch_stats_32["fully_dead_batches"]

    # But even partially active batches can benefit: if only some tiles are active,
    # the data loading still happens but processing can be skipped for dead tiles.
    # The current backward already does warp-level processing skip, so the NEW
    # benefit is in data loading skip for fully dead batches.

    return {
        "scene": state["scene"],
        "ALREADY_REMOVED_BY_CURRENT_LAST_ID": {
            "description": "The current backward already skips PROCESSING for entries after bin_final per pixel (line 220) and uses warp_bin_final for warp-level batch processing skip (line 217). However, DATA LOADING still happens unconditionally for ALL batches (lines 198-212).",
            "already_skipped_entries": int(already_skipped_entries),
            "total_entries": int(total_entries),
            "already_skipped_fraction": already_skipped_entries / total_entries if total_entries > 0 else 0,
            "note": "These entries are skipped in PROCESSING but their data is still LOADED. The processing skip is warp-level (warp_bin_final) and per-pixel (bin_final).",
        },
        "NEW_HIERARCHICAL_REMOVABLE_WORK": {
            "description": "NEW work that hierarchical pruning could remove beyond what the current backward already skips.",
            "1024G_batch_level": {
                "fully_dead_batches": fully_dead_1024,
                "fully_dead_batch_fraction": fully_dead_1024 / batch_stats_1024["total_batches"] if batch_stats_1024["total_batches"] > 0 else 0,
                "dead_batch_tile_groups": new_dead_1024,
                "dead_batch_tile_fraction": new_dead_1024 / batch_stats_1024["total_batch_tile_groups"] if batch_stats_1024["total_batch_tile_groups"] > 0 else 0,
                "removable_sorted_id_loads": fully_dead_1024 * BATCH_1024,
                "removable_attribute_loads_estimate": fully_dead_1024 * BATCH_1024,  # upper bound
                "note": "Fully dead batches can have ALL data loading skipped (sorted IDs + attributes). Partially active batches can skip data loading for dead tiles within the batch, but this requires per-tile masking in the loading phase.",
            },
            "32G_minibatch_level": {
                "fully_dead_minibatches": fully_dead_32,
                "fully_dead_minibatch_fraction": fully_dead_32 / batch_stats_32["total_batches"] if batch_stats_32["total_batches"] > 0 else 0,
                "dead_minibatch_tile_groups": new_dead_32,
                "dead_minibatch_tile_fraction": new_dead_32 / batch_stats_32["total_batch_tile_groups"] if batch_stats_32["total_batch_tile_groups"] > 0 else 0,
                "removable_sorted_id_loads": fully_dead_32 * BATCH_32,
                "removable_attribute_loads_estimate": fully_dead_32 * BATCH_32,
                "note": "32-G mini-batches provide finer granularity. The additional dead groups beyond 1024-G represent the gain from finer pruning.",
            },
            "incremental_gain_32G_over_1024G": {
                "additional_dead_batch_tile_groups": new_dead_32 - new_dead_1024,
                "description": "The additional dead groups from 32-G granularity vs 1024-G. This is the marginal benefit of finer pruning.",
            },
        },
        "WARNING": "Load counts do NOT directly translate to runtime speedup. The backward is likely memory-bandwidth bound, so reducing loads may help, but the relationship is non-linear. Kernel launch overhead, shared memory bank conflicts, and warp scheduling all affect actual speedup.",
    }


def section7_pixel_alive_stats(state, macro_rep, batch_size=32):
    """Section 7: Pixel-level active masks for selected representative batches."""
    tw, th = state["tw"], state["tw"]
    H, W = state["height"], state["width"]
    last_ids = state["last_ids"]
    offs = state["offs"]
    flat_len = len(state["flat"])
    off_flat = np.append(offs.ravel(), flat_len)

    # For selected tiles, measure how many pixels are still "alive" (have remaining work)
    # at different points in the tile's sorted list.
    # A pixel is "alive" at position p if last_id >= tile_start + p (i.e., it can still
    # reach Gaussians at position p or later).

    # Sample representative tiles: pick tiles with various list lengths
    tile_list_lengths = np.diff(off_flat)
    active_tiles = np.where(tile_list_lengths > 0)[0]

    if len(active_tiles) == 0:
        return {"scene": state["scene"], "note": "No active tiles"}

    # Sample up to 100 tiles, stratified by list length
    n_sample = min(100, len(active_tiles))
    indices = np.linspace(0, len(active_tiles) - 1, n_sample).astype(int)
    sampled_tiles = active_tiles[indices]

    # For each sampled tile, compute pixel alive counts at different depths
    alive_fractions = []
    for tid in sampled_tiles:
        tlen = int(tile_list_lengths[tid])
        tstart = int(off_flat[tid])
        ty = tid // tw
        tx = tid % tw

        # Get last_ids for pixels in this tile
        py0, py1 = ty * TILE_SIZE, min((ty + 1) * TILE_SIZE, H)
        px0, px1 = tx * TILE_SIZE, min((tx + 1) * TILE_SIZE, W)
        tile_last_ids = []
        for py in range(py0, py1):
            for px in range(px0, px1):
                tile_last_ids.append(int(last_ids[py, px]))

        tile_last_ids = np.array(tile_last_ids)
        n_pixels = len(tile_last_ids)

        # At position p in the tile list, a pixel is alive if last_id >= tstart + p
        # Compute alive fraction at p = 0, tlen/4, tlen/2, 3*tlen/4, tlen-1
        checkpoints = [0, tlen // 4, tlen // 2, 3 * tlen // 4, tlen - 1]
        for p in checkpoints:
            if p < 0 or p >= tlen:
                continue
            alive = np.sum(tile_last_ids >= tstart + p)
            alive_fractions.append({
                "tile_id": int(tid),
                "tile_list_length": tlen,
                "position": p,
                "position_fraction": p / tlen if tlen > 0 else 0,
                "alive_pixels": int(alive),
                "total_pixels": n_pixels,
                "alive_fraction": alive / n_pixels if n_pixels > 0 else 0,
            })

    # Aggregate: at each position fraction, what's the mean alive fraction?
    position_fracs = {}
    for af in alive_fractions:
        key = f"{af['position_fraction']:.2f}"
        if key not in position_fracs:
            position_fracs[key] = []
        position_fracs[key].append(af["alive_fraction"])

    alive_summary = {}
    for key, vals in sorted(position_fracs.items()):
        alive_summary[key] = {
            "mean_alive_fraction": float(np.mean(vals)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "n_tiles": len(vals),
        }

    return {
        "scene": state["scene"],
        "n_sampled_tiles": n_sample,
        "alive_fraction_by_position": alive_summary,
        "detail": alive_fractions[:50],  # First 50 for inspection
        "note": "A pixel is 'alive' at position p if last_id >= tile_start + p. This measures how many pixels still have remaining work at different depths in the tile's sorted list. A batch commonly having only a small subset of pixels alive could enable a second-level exact pruning mechanism.",
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
    print(f"=== H5-0: Last-ID Guided Hierarchical Backward Pruning Oracle ===", flush=True)
    print(f"run_id={run_id} GPU={args.gpu}", flush=True)

    bootstrap(args.source, args.core_so)
    print("Bootstrap complete.", flush=True)

    # Section 1: last_ids semantic contract (source-verified)
    print("\n=== Section 1: last_ids Semantic Contract ===", flush=True)
    last_id_semantics = {
        "verdict": "LAST_ID_ABSOLUTE",
        "description": "last_ids[pix_id] stores the ABSOLUTE position in the global flatten_ids array (sorted tile-Gaussian intersection list) of the last Gaussian that contributed to this pixel. It is NOT a tile-local rank.",
        "evidence": {
            "forward_writer": {
                "file": "experiments/r6/_mxsrc/RasterizeToPixels3DGSFwd.cu",
                "line": 166,
                "code": "cur_idx = batch_start + t; // where batch_start = range_start + block_size * b, range_start = tile_offsets[tile_id]",
                "write_line": 186,
                "write_code": "last_ids[pix_id] = static_cast<int32_t>(cur_idx);",
                "comment": "line 185: 'index in bin of last gaussian in this pixel' — 'bin' means the tile's range within the global flatten_ids array, but the stored value is the ABSOLUTE index, not a 0-based tile-local rank.",
            },
            "autograd_adapter": {
                "file": ".build_tmp/gaussian_inference.py",
                "forward_return_line": 2159,
                "save_line": 2200,
                "backward_pass_line": 2339,
                "code": "last_ids=last_ids_f",
                "note": "last_ids passes through UNCHANGED from forward output to backward input. No transformation.",
            },
            "backward_reader": {
                "file": ".build_tmp/HigsNativeBackward.cu",
                "read_line": 162,
                "read_code": "const int32_t bin_final = inside ? last_ids[pix_id] : 0;",
                "warp_reduce_line": 188,
                "warp_reduce_code": "const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());",
                "batch_end_line": 196,
                "batch_end_code": "const int32_t batch_end = range_end - 1 - (int32_t)(block_size * b); // ABSOLUTE position in flatten_ids",
                "warp_skip_line": 217,
                "warp_skip_code": "for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t) // warp-level skip",
                "per_pixel_skip_line": 220,
                "per_pixel_skip_code": "if(batch_end - (int32_t)t > bin_final) { valid = 0; } // per-pixel skip: compares ABSOLUTE positions",
                "note": "The backward compares batch_end (absolute position) against bin_final (also absolute). This confirms the value is ABSOLUTE, not tile-local.",
            },
            "base_gsplat_forward": {
                "file": ".build_tmp/r6b_b0/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu",
                "line": 164,
                "code": "cur_idx = batch_start + t; // same absolute pattern",
                "write_line": 184,
            },
            "base_gsplat_backward": {
                "file": ".build_tmp/r6b_b0/gsplat/cuda/csrc/rasterize_to_pixels_bwd.cu",
                "line": 112,
                "code": "const int32_t bin_final = inside ? last_ids[pix_id] : 0; // same absolute pattern",
            },
        },
        "current_backward_skip_mechanism": {
            "level_1_warp": "warp_bin_final = max(bin_final) over warp. Processing loop starts at max(0, batch_end - warp_bin_final). If warp_bin_final << batch_end, most of the batch's processing is skipped.",
            "level_2_per_pixel": "if (batch_end - t > bin_final) valid = 0. Each pixel skips processing for entries after its own last_id.",
            "critical_gap": "DATA LOADING (lines 198-212) happens UNCONDITIONALLY for ALL batches. The skip only affects processing, not data loading. This is the NEW optimization opportunity: skip data loading for provably dead batches.",
            "no_block_level_break": "Unlike the forward (which has __syncthreads_count(done) >= block_size break), the backward has NO block-level early break. All batches are loaded even if the entire tile is done.",
        },
    }
    (out_dir / "last_id_semantics.json").write_text(json.dumps(last_id_semantics, indent=2))
    print(f"  Verdict: LAST_ID_ABSOLUTE", flush=True)

    # Process each scene
    all_tail_stats = {}
    all_batch_1024 = {}
    all_batch_32 = {}
    all_removable = {}
    all_pixel_alive = {}
    scene_summaries = {}

    for scene in ["room", "bicycle", "garden"]:
        print(f"\n=== Processing {scene} ===", flush=True)

        # Section 2: Capture exact real-scene state
        print(f"  Capturing forward state...", flush=True)
        state = make_fixture_and_render(scene, args.max_long_side, device)
        print(f"  {state['width']}x{state['height']}, N_vis={state['n_vis']}, "
              f"n_isects={len(state['flat'])}, tiles={state['tw']}x{state['th']}", flush=True)

        # Compute macro representation
        print(f"  Computing macro representation...", flush=True)
        macro_rep = compute_macro_representation(state)
        print(f"  Macro: {macro_rep['mw']}x{macro_rep['mh']} macro tiles, "
              f"{macro_rep['n_macro_entries']} entries, {macro_rep['n_fine_pairs']} fine pairs", flush=True)

        scene_summaries[scene] = {
            "scene": scene,
            "width": state["width"], "height": state["height"],
            "tw": state["tw"], "th": state["th"],
            "n_vis": state["n_vis"],
            "n_isects": len(state["flat"]),
            "n_macro_tiles": macro_rep["mw"] * macro_rep["mh"],
            "n_macro_entries": macro_rep["n_macro_entries"],
            "n_fine_pairs": macro_rep["n_fine_pairs"],
            "representation_compression": macro_rep["n_fine_pairs"] / macro_rep["n_macro_entries"] if macro_rep["n_macro_entries"] > 0 else 0,
        }

        # Section 3: Per-pixel tail pruning
        print(f"  Section 3: Per-pixel tail pruning...", flush=True)
        tail_stats = section3_per_pixel_tail_stats(state)
        all_tail_stats[scene] = tail_stats
        print(f"  tile_list mean={tail_stats['tile_list_length']['mean']:.1f} "
              f"reachable mean={tail_stats['reachable_length']['mean']:.1f} "
              f"tail_skip mean={tail_stats['tail_skip_fraction']['mean']:.3f}", flush=True)

        # Section 4: 1024-G batch level
        print(f"  Section 4: 1024-G batch level...", flush=True)
        batch_1024 = section4_batch_level(state, macro_rep, BATCH_1024)
        all_batch_1024[scene] = batch_1024
        print(f"  batches={batch_1024['total_batches']} dead={batch_1024['fully_dead_batches']} "
              f"batch_tile_dead={batch_1024['batch_tile_dead_fraction']:.3f}", flush=True)

        # Section 5: 32-G mini-batch level
        print(f"  Section 5: 32-G mini-batch level...", flush=True)
        batch_32 = section4_batch_level(state, macro_rep, BATCH_32)
        all_batch_32[scene] = batch_32
        print(f"  minibatches={batch_32['total_batches']} dead={batch_32['fully_dead_batches']} "
              f"minibatch_tile_dead={batch_32['batch_tile_dead_fraction']:.3f}", flush=True)

        # Section 6: Removable work
        print(f"  Section 6: Removable work...", flush=True)
        removable = section6_removable_work(state, macro_rep, batch_1024, batch_32)
        all_removable[scene] = removable
        print(f"  already_skipped={removable['ALREADY_REMOVED_BY_CURRENT_LAST_ID']['already_skipped_fraction']:.3f} "
              f"new_1024G_dead={removable['NEW_HIERARCHICAL_REMOVABLE_WORK']['1024G_batch_level']['fully_dead_batch_fraction']:.3f} "
              f"new_32G_dead={removable['NEW_HIERARCHICAL_REMOVABLE_WORK']['32G_minibatch_level']['fully_dead_minibatch_fraction']:.3f}", flush=True)

        # Section 7: Pixel alive stats
        print(f"  Section 7: Pixel alive stats...", flush=True)
        pixel_alive = section7_pixel_alive_stats(state, macro_rep)
        all_pixel_alive[scene] = pixel_alive

    # Write all artifacts
    print("\n=== Writing artifacts ===", flush=True)
    (out_dir / "scene_state_summary.json").write_text(json.dumps(scene_summaries, indent=2))
    (out_dir / "per_pixel_tail_stats.json").write_text(json.dumps(all_tail_stats, indent=2))
    (out_dir / "batch_active_masks.json").write_text(json.dumps(all_batch_1024, indent=2))
    (out_dir / "minibatch_active_masks.json").write_text(json.dumps(all_batch_32, indent=2))
    (out_dir / "removable_work.json").write_text(json.dumps(all_removable, indent=2))
    (out_dir / "pixel_alive_stats.json").write_text(json.dumps(all_pixel_alive, indent=2))

    # Section 9: Gate decision
    print("\n=== Section 9: Gate Decision ===", flush=True)
    # STRONG: >= 25% batch×tile groups dead on at least 2/3 scenes
    # MARGINAL: 10-25% structural work removal
    # WEAK: <10%

    dead_fracs_1024 = [all_batch_1024[s]["batch_tile_dead_fraction"] for s in ["room", "bicycle", "garden"]]
    dead_fracs_32 = [all_batch_32[s]["batch_tile_dead_fraction"] for s in ["room", "bicycle", "garden"]]
    new_dead_1024 = [all_removable[s]["NEW_HIERARCHICAL_REMOVABLE_WORK"]["1024G_batch_level"]["fully_dead_batch_fraction"] for s in ["room", "bicycle", "garden"]]
    new_dead_32 = [all_removable[s]["NEW_HIERARCHICAL_REMOVABLE_WORK"]["32G_minibatch_level"]["fully_dead_minibatch_fraction"] for s in ["room", "bicycle", "garden"]]

    n_strong_1024 = sum(1 for d in dead_fracs_1024 if d >= 0.25)
    n_strong_32 = sum(1 for d in dead_fracs_32 if d >= 0.25)
    n_marginal_1024 = sum(1 for d in dead_fracs_1024 if d >= 0.10)
    n_marginal_32 = sum(1 for d in dead_fracs_32 if d >= 0.10)

    avg_dead_1024 = float(np.mean(dead_fracs_1024))
    avg_dead_32 = float(np.mean(dead_fracs_32))
    avg_new_dead_1024 = float(np.mean(new_dead_1024))
    avg_new_dead_32 = float(np.mean(new_dead_32))

    # Gate logic
    if n_strong_1024 >= 2 or n_strong_32 >= 2:
        gate = "LASTID_PRUNING_STRONG"
    elif n_marginal_1024 >= 2 or n_marginal_32 >= 2:
        gate = "LASTID_PRUNING_MARGINAL"
    else:
        gate = "LASTID_PRUNING_WEAK"

    analysis = {
        "gate": gate,
        "batch_tile_dead_fractions_1024G": {s: all_batch_1024[s]["batch_tile_dead_fraction"] for s in ["room", "bicycle", "garden"]},
        "batch_tile_dead_fractions_32G": {s: all_batch_32[s]["batch_tile_dead_fraction"] for s in ["room", "bicycle", "garden"]},
        "fully_dead_batch_fractions_1024G": {s: all_removable[s]["NEW_HIERARCHICAL_REMOVABLE_WORK"]["1024G_batch_level"]["fully_dead_batch_fraction"] for s in ["room", "bicycle", "garden"]},
        "fully_dead_minibatch_fractions_32G": {s: all_removable[s]["NEW_HIERARCHICAL_REMOVABLE_WORK"]["32G_minibatch_level"]["fully_dead_minibatch_fraction"] for s in ["room", "bicycle", "garden"]},
        "avg_batch_tile_dead_1024G": avg_dead_1024,
        "avg_batch_tile_dead_32G": avg_dead_32,
        "avg_fully_dead_batch_1024G": avg_new_dead_1024,
        "avg_fully_dead_minibatch_32G": avg_new_dead_32,
        "n_scenes_strong_1024G": n_strong_1024,
        "n_scenes_strong_32G": n_strong_32,
        "n_scenes_marginal_1024G": n_marginal_1024,
        "n_scenes_marginal_32G": n_marginal_32,
        "oracle_timing": "NOT_AVAILABLE — Section 8 oracle timing skipped. Requires building a modified backward kernel with provably-dead batch skip. Structural analysis determines whether this investment is warranted.",
        "correctness": "NOT_APPLICABLE — No kernel was modified. All analysis is based on forward-computed last_ids and structural projection. Gradients are unchanged.",
        "gate_criteria": {
            "STRONG": ">= 25% batch×tile groups dead on at least 2/3 scenes",
            "MARGINAL": "10-25% structural work removal",
            "WEAK": "<10% new hierarchical work removable",
        },
    }
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (out_dir / "correctness.json").write_text(json.dumps({
        "status": "NOT_APPLICABLE",
        "description": "No kernel was modified. All analysis is structural, based on forward-computed last_ids and macro hierarchy projection. Gradients are unchanged. Oracle timing (Section 8) was not performed, so no gradient comparison is needed.",
    }, indent=2))

    # Write empty oracle_timing.csv (Section 8 not performed)
    (out_dir / "oracle_timing.csv").write_text("scene,variant,median_ms,mean_ms,n\n")

    # Provenance
    core_so_sha = hashlib.sha256(Path(args.core_so).read_bytes()).hexdigest() if Path(args.core_so).exists() else "N/A"
    provenance = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "gpu": f"cuda:{args.gpu}",
        "source": args.source,
        "core_so": args.core_so,
        "core_so_sha256": core_so_sha,
        "max_long_side": args.max_long_side,
        "method": "CPU structural analysis using GPU-computed forward state (last_ids) + CPU macro representation",
        "section_8_oracle_timing": "SKIPPED — structural analysis determines whether kernel modification investment is warranted",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"\n=== Decision: {gate} ===", flush=True)
    print(f"  1024G batch_tile_dead: {dead_fracs_1024}", flush=True)
    print(f"  32G batch_tile_dead: {dead_fracs_32}", flush=True)
    print(f"  1024G fully_dead_batch: {new_dead_1024}", flush=True)
    print(f"  32G fully_dead_minibatch: {new_dead_32}", flush=True)


if __name__ == "__main__":
    main()
