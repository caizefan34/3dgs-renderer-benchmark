#!/usr/bin/env python3
"""H2-BWD-0 Structural Analysis: Macro-tile pair oracle, distributions, batch counts.
Fully vectorized with numpy. Runs on mx in higs-13scene-env.

Computes sections 1-6, 8-9 of the H2-BWD-0 specification.
"""
import argparse, json, math, os, csv
import numpy as np
import torch
from plyfile import PlyData

SH_DEGREE = 3
K_SH = (SH_DEGREE + 1) ** 2  # 16
FUSED_MACRO_TILE_WIDTH = 8
FUSED_MACRO_TILE_HEIGHT = 4
FUSED_GAUSS_BATCH_SIZE = 1024
TILE_SIZE = 16

def load_ply_scene(ply_path, device):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    N = len(v)
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), dtype=torch.float32, device=device)
    quats = torch.tensor(np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]), dtype=torch.float32, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]), dtype=torch.float32, device=device))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], dtype=torch.float32, device=device))
    f_dc = torch.tensor(np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]), dtype=torch.float32, device=device)
    n_rest = 3 * (K_SH - 1)
    f_rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], dtype=torch.float32, device=device) for i in range(n_rest)], dim=1)
    f_rest = f_rest.reshape(N, 3, K_SH - 1).permute(0, 2, 1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = f_dc
    sh[:, 1:] = f_rest
    return means, quats, scales, opacities, sh

def load_cameras(cams_path, width, height, device):
    with open(cams_path) as f:
        cams = json.load(f)
    viewmats, Ks = [], []
    for c in cams:
        R = np.asarray(c["rotation"], dtype=np.float64)
        p = np.asarray(c["position"], dtype=np.float64)
        Rw2c = R.T
        vm = np.eye(4)
        vm[:3, :3] = Rw2c
        vm[:3, 3] = -Rw2c @ p
        scale = width / float(c["width"])
        K = np.array([[float(c["fx"]) * scale, 0.0, (width - 1) / 2.0],
                      [0.0, float(c["fy"]) * scale, (height - 1) / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        viewmats.append(torch.tensor(vm, dtype=torch.float32, device=device))
        Ks.append(torch.tensor(K, dtype=torch.float32, device=device))
    return torch.stack(viewmats).unsqueeze(0), torch.stack(Ks).unsqueeze(0)

def run_forward_get_isects(args, device):
    """Run B2 forward to get tile_offsets, flatten_ids, means2d, radii."""
    scene_configs = {
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
    }
    cfg = scene_configs[args.scene]
    nw, nh = cfg["native_w"], cfg["native_h"]
    if args.max_long_side > 0 and max(nw, nh) > args.max_long_side:
        scale = args.max_long_side / max(nw, nh)
        width = max(1, int(round(nw * scale)))
        height = max(1, int(round(nh * scale)))
    else:
        width, height = nw, nh

    means, quats, scales, opacities, sh = load_ply_scene(cfg["ply"], device)
    N_total = len(means)
    viewmats_all, Ks_all = load_cameras(cfg["cams"], width, height, device)
    vm = viewmats_all[:, [args.cam_idx]]
    K = Ks_all[:, [args.cam_idx]]
    colors = sh

    tile_width = math.ceil(width / TILE_SIZE)
    tile_height = math.ceil(height / TILE_SIZE)
    n_tiles = tile_width * tile_height

    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
    from gsplat.rendering import _maybe_evaluate_sh
    from gsplat.experimental.render.functional.gaussian_inference import (
        _cull_gaussians_batched, _gather_visible_native,
    )

    with torch.no_grad():
        visible_ids, _, culling_ratio = _cull_gaussians_batched(
            means, quats, scales, vm, K, width, height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, camera_model="pinhole",
        )
    N_visible = visible_ids.numel()

    with torch.no_grad():
        v_means, v_quats, v_scales, v_opacities, v_colors = _gather_visible_native(
            means, quats, scales, opacities, colors, visible_ids,
        )

    v_means_b = v_means.unsqueeze(0).contiguous()
    v_quats_b = v_quats.unsqueeze(0).contiguous()
    v_scales_b = v_scales.unsqueeze(0).contiguous()
    v_opacities_b = v_opacities.unsqueeze(0).contiguous()
    v_colors_input = v_colors.unsqueeze(0) if v_colors.dim() == 2 else v_colors
    C = vm.shape[-3]
    opacities_bc = torch.broadcast_to(v_opacities_b[..., None, :], (1, C, N_visible)).contiguous()

    with torch.no_grad():
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=v_means_b, covars=None, quats=v_quats_b, scales=v_scales_b,
            viewmats=vm, Ks=K, width=width, height=height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
            calc_compensations=False, camera_model="pinhole",
        )

    with torch.no_grad():
        colors_eval = _maybe_evaluate_sh(
            SH_DEGREE, v_colors_input, v_means_b, radii, vm, (1,), C, N_visible, True,
        ).contiguous()

    with torch.no_grad():
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, TILE_SIZE, tile_width, tile_height,
            packed=False, n_images=C, image_ids=None, gaussian_ids=None,
            conics=conics, opacities=opacities_bc,
        )
        isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height).reshape((1, C, tile_height, tile_width))

    n_isects = isect_ids.numel()
    return {
        "width": width, "height": height,
        "tile_width": tile_width, "tile_height": tile_height, "n_tiles": n_tiles,
        "N_total": N_total, "N_visible": N_visible, "culling_ratio": culling_ratio,
        "n_isects": n_isects,
        "flatten_ids": flatten_ids.cpu().numpy().astype(np.int64),
        "isect_offsets": isect_offsets.cpu().numpy().astype(np.int32),
        "means2d": means2d.cpu().numpy(),  # [1, C, N_v, 2]
        "radii": radii.cpu().numpy(),      # [1, C, N_v, 2]
    }

def compute_structural_stats(fwd, scene, cam_idx, out_dir):
    """All sections 1-6, 8-9, fully vectorized."""
    tile_width = fwd["tile_width"]
    tile_height = fwd["tile_height"]
    n_tiles = fwd["n_tiles"]
    n_isects = fwd["n_isects"]
    N_visible = fwd["N_visible"]
    flatten_ids = fwd["flatten_ids"]
    isect_offsets = fwd["isect_offsets"]
    means2d = fwd["means2d"][0, 0]  # [N_v, 2]
    radii = fwd["radii"][0, 0]      # [N_v, 2]

    macro_width = math.ceil(tile_width / FUSED_MACRO_TILE_WIDTH)
    macro_height = math.ceil(tile_height / FUSED_MACRO_TILE_HEIGHT)
    n_macro_tiles = macro_width * macro_height

    print(f"[geom] tiles={tile_width}x{tile_height}={n_tiles}, macro={macro_width}x{macro_height}={n_macro_tiles}")
    print(f"[geom] fine_per_macro={FUSED_MACRO_TILE_WIDTH}x{FUSED_MACRO_TILE_HEIGHT}={FUSED_MACRO_TILE_WIDTH*FUSED_MACRO_TILE_HEIGHT}")
    print(f"[isect] n_isects={n_isects}, N_visible={N_visible}")

    # ---- Build per-intersection arrays (vectorized) ----
    # isect_offsets is [1, C, tile_h, tile_w], C=1 → [tile_h, tile_w]
    offsets_flat = isect_offsets.reshape(-1)  # [n_tiles]
    offsets_ext = np.append(offsets_flat, n_isects)

    # For each intersection, find which tile it belongs to
    # Vectorized: build tile_id per intersection using searchsorted
    tile_ids_per_isect = np.searchsorted(offsets_ext, np.arange(n_isects), side="right") - 1
    tile_ids_per_isect = np.clip(tile_ids_per_isect, 0, n_tiles - 1).astype(np.int32)

    gaussian_ids_per_isect = flatten_ids.astype(np.int32)

    # tile coordinates
    tile_x_arr = (tile_ids_per_isect % tile_width).astype(np.int32)
    tile_y_arr = (tile_ids_per_isect // tile_width).astype(np.int32)

    # macro coordinates
    macro_x_arr = tile_x_arr // FUSED_MACRO_TILE_WIDTH
    macro_y_arr = tile_y_arr // FUSED_MACRO_TILE_HEIGHT
    macro_ids_per_isect = (macro_y_arr * macro_width + macro_x_arr).astype(np.int32)

    # ============================================================
    # SECTION 1: Macro-tile pair oracle
    # ============================================================
    print("\n===== Section 1: Macro-tile pair oracle =====")
    macro_gauss_keys = macro_ids_per_isect.astype(np.int64) * N_visible + gaussian_ids_per_isect.astype(np.int64)
    unique_macro_gauss = np.unique(macro_gauss_keys)
    N_macro_gaussian_pairs = len(unique_macro_gauss)
    R_macro = n_isects / N_macro_gaussian_pairs if N_macro_gaussian_pairs > 0 else 0

    print(f"N_tile_gaussian_pairs = {n_isects}")
    print(f"N_macro_gaussian_pairs = {N_macro_gaussian_pairs}")
    print(f"R_macro = {R_macro:.4f}")

    # ============================================================
    # SECTION 2: Fine-tile reuse distribution inside macro tiles
    # ============================================================
    print("\n===== Section 2: Fine-tile reuse distribution =====")
    # For each unique (macro_id, gaussian_id), count unique fine tiles
    # Vectorized: build (macro_gauss_key, tile_id) pairs, sort, group, count unique tile_ids
    pair_tile = np.stack([macro_gauss_keys, tile_ids_per_isect.astype(np.int64)], axis=1)
    # Sort by macro_gauss_key then tile_id
    sort_idx = np.lexsort((pair_tile[:, 1], pair_tile[:, 0]))
    sorted_pairs = pair_tile[sort_idx]

    # Find group boundaries by macro_gauss_key
    unique_keys, start_idx = np.unique(sorted_pairs[:, 0], return_index=True)
    n_unique_keys = len(unique_keys)
    end_idx = np.append(start_idx[1:], len(sorted_pairs))

    # For each group, count unique tile_ids
    # Within a sorted group (by key, then tile_id), count changes in tile_id
    # A new unique tile is when tile_id differs from previous OR it's the first in the group
    tile_ids_sorted = sorted_pairs[:, 1]
    key_starts = start_idx
    is_first_in_group = np.zeros(len(sorted_pairs), dtype=bool)
    is_first_in_group[key_starts] = True
    is_new_tile = (tile_ids_sorted != np.roll(tile_ids_sorted, 1)) | is_first_in_group
    # Count per group
    group_ids = np.repeat(np.arange(n_unique_keys), end_idx - start_idx)
    fine_tile_counts = np.bincount(group_ids, weights=is_new_tile.astype(np.int64)).astype(np.int32)

    mean_ft = float(np.mean(fine_tile_counts))
    p50_ft = float(np.percentile(fine_tile_counts, 50))
    p75_ft = float(np.percentile(fine_tile_counts, 75))
    p90_ft = float(np.percentile(fine_tile_counts, 90))
    p95_ft = float(np.percentile(fine_tile_counts, 95))
    p99_ft = float(np.percentile(fine_tile_counts, 99))
    max_ft = int(np.max(fine_tile_counts))

    print(f"Fine-tile reuse: mean={mean_ft:.2f} p50={p50_ft} p75={p75_ft} p90={p90_ft} p95={p95_ft} p99={p99_ft} max={max_ft}")

    hist_bins = [1, 2, "3-4", "5-8", "9-16", "17-24", "25-32"]
    hist_counts = [
        int(np.sum(fine_tile_counts == 1)),
        int(np.sum(fine_tile_counts == 2)),
        int(np.sum((fine_tile_counts >= 3) & (fine_tile_counts <= 4))),
        int(np.sum((fine_tile_counts >= 5) & (fine_tile_counts <= 8))),
        int(np.sum((fine_tile_counts >= 9) & (fine_tile_counts <= 16))),
        int(np.sum((fine_tile_counts >= 17) & (fine_tile_counts <= 24))),
        int(np.sum((fine_tile_counts >= 25) & (fine_tile_counts <= 32))),
    ]
    print(f"Histogram: {dict(zip(hist_bins, hist_counts))}")

    # ============================================================
    # SECTION 3: Macro-tile workload distribution
    # ============================================================
    print("\n===== Section 3: Macro-tile workload distribution =====")
    # For each macro tile: N_unique_gaussians, N_fine_tile_gaussian_pairs, N_active_fine_tiles
    # Vectorized using bincount
    # N_unique_gaussians: count unique gaussian_ids per macro_id
    # Use the unique_macro_gauss keys to extract macro_ids
    macro_ids_unique = (unique_macro_gauss // N_visible).astype(np.int32)
    n_unique_gauss_per_macro = np.bincount(macro_ids_unique, minlength=n_macro_tiles)

    # N_fine_tile_gaussian_pairs: count intersections per macro_id
    n_pairs_per_macro = np.bincount(macro_ids_per_isect, minlength=n_macro_tiles)

    # N_active_fine_tiles: count unique tile_ids per macro_id
    # Build (macro_id, tile_id) pairs, unique them, then bincount
    macro_tile_pairs = np.stack([macro_ids_per_isect.astype(np.int64), tile_ids_per_isect.astype(np.int64)], axis=1)
    unique_macro_tile = np.unique(macro_tile_pairs, axis=0)
    n_active_tiles_per_macro = np.bincount(unique_macro_tile[:, 0], minlength=n_macro_tiles)

    # Only consider active macro tiles (with at least 1 Gaussian)
    active_mask = n_unique_gauss_per_macro > 0
    n_active_macros = int(np.sum(active_mask))

    macro_unique_gauss = n_unique_gauss_per_macro[active_mask]
    macro_pairs = n_pairs_per_macro[active_mask]
    macro_active_tiles = n_active_tiles_per_macro[active_mask]

    def print_stats(name, arr):
        print(f"  {name}: mean={np.mean(arr):.1f} p50={np.percentile(arr,50):.0f} p90={np.percentile(arr,90):.0f} "
              f"p95={np.percentile(arr,95):.0f} p99={np.percentile(arr,99):.0f} max={np.max(arr)}")

    print_stats("N_unique_gaussians", macro_unique_gauss)
    print_stats("N_fine_tile_gaussian_pairs", macro_pairs)
    print_stats("N_active_fine_tiles", macro_active_tiles)
    print(f"  N_macro_tiles_active = {n_active_macros} / {n_macro_tiles}")

    # 1024-batch distribution
    n_batches_per_macro = np.ceil(macro_unique_gauss / FUSED_GAUSS_BATCH_SIZE).astype(np.int32)
    print_stats("N_1024_batches", n_batches_per_macro)
    batch_hist = {}
    for b in n_batches_per_macro:
        batch_hist[int(b)] = batch_hist.get(int(b), 0) + 1
    print(f"  batch_hist = {dict(sorted(batch_hist.items()))}")

    # ============================================================
    # SECTION 4: 32-Gaussian mini-batch feasibility
    # ============================================================
    print("\n===== Section 4: 32-Gaussian mini-batch feasibility =====")
    grad_state_bytes = 32 * 9 * 4  # 1152
    # Metadata: 32 Gaussian IDs (int32) = 128, tile occupancy bitmask (32 bits = 4 bytes per warp, 4 warps = 16), reduction scratch (9*4=36)
    metadata_bytes = 128 + 16 + 36
    total_smem_estimate = grad_state_bytes + metadata_bytes
    # Also estimate for the full 1024-Gaussian batch case (for comparison)
    full_batch_smem = 1024 * 9 * 4  # 36864 bytes if all in shared memory

    print(f"  Per-mini-batch gradient state: {grad_state_bytes} bytes (32 * 9 * 4)")
    print(f"  Per-mini-batch metadata: {metadata_bytes} bytes")
    print(f"  Per-mini-batch total smem: {total_smem_estimate} bytes")
    print(f"  Full 1024-batch gradient state (if all in smem): {full_batch_smem} bytes")

    # Occupancy estimate for mini-batch approach
    regs_per_thread = 64
    threads_per_block = 128
    regs_per_block = regs_per_thread * threads_per_block
    max_blocks_regs = 65536 // regs_per_block
    max_blocks_smem = 164 * 1024 // max(total_smem_estimate, 1)
    max_blocks_threads = 2048 // threads_per_block
    est_blocks_per_sm = min(max_blocks_regs, max_blocks_smem, max_blocks_threads)
    print(f"  Mini-batch est blocks/SM: {est_blocks_per_sm} (regs={max_blocks_regs}, smem={max_blocks_smem}, threads={max_blocks_threads})")

    if total_smem_estimate <= 16000:
        feasibility = "SAFE"
    elif total_smem_estimate <= 48000:
        feasibility = "TIGHT"
    else:
        feasibility = "UNSAFE"
    print(f"  Mini-batch feasibility: {feasibility}")

    # ============================================================
    # SECTION 5: Current cross-warp scatter multiplicity
    # ============================================================
    print("\n===== Section 5: Current cross-warp scatter multiplicity =====")
    # Kernel: higs_blend_bwd_px_kernel<3,2>
    # Block: (16, 8, 1) = 128 threads = 4 warps
    # threadIdx.x = 0..15 (tx), threadIdx.y = 0..7 (ty)
    # Warp w (w=0..3): ty = 2w, 2w+1
    # Pixel rows per warp (PX=2, PX_ROWS=8):
    #   ty=q: pixel rows i0+q and i0+q+8
    #   Warp 0 (ty=0,1): rows {0,1,8,9}
    #   Warp 1 (ty=2,3): rows {2,3,10,11}
    #   Warp 2 (ty=4,5): rows {4,5,12,13}
    #   Warp 3 (ty=6,7): rows {6,7,14,15}
    # Each warp covers ALL 16 x-columns (tx=0..15)
    # Scatter multiplicity per intersection = number of warps whose pixel rows
    # overlap the Gaussian's y-extent within the tile (clamped to [0,15])

    # Get per-intersection Gaussian center y and radius y
    gy = means2d[gaussian_ids_per_isect, 1]  # global y center
    ry = radii[gaussian_ids_per_isect, 1]    # y radius

    # Tile origin y
    tile_origin_y = tile_y_arr * TILE_SIZE

    # Local y-extent within tile
    y_min_local = np.maximum(0.0, gy - ry - tile_origin_y)
    y_max_local = np.minimum(float(TILE_SIZE - 1), gy + ry - tile_origin_y)

    # Warp row sets: warp w covers rows {2w, 2w+1, 2w+8, 2w+9}
    # Overlap with [y_min, y_max] iff:
    #   (y_min <= 2w+1 AND y_max >= 2w) OR (y_min <= 2w+9 AND y_max >= 2w+8)
    n_warps_per_isect = np.zeros(n_isects, dtype=np.int32)
    for w in range(4):
        lo1, hi1 = 2*w, 2*w+1      # first row pair
        lo2, hi2 = 2*w+8, 2*w+9    # second row pair
        overlap = ((y_min_local <= hi1) & (y_max_local >= lo1)) | \
                  ((y_min_local <= hi2) & (y_max_local >= lo2))
        n_warps_per_isect += overlap.astype(np.int32)

    N_current_scatter_groups = int(np.sum(n_warps_per_isect))
    R_crosswarp = N_current_scatter_groups / n_isects if n_isects > 0 else 0
    R_total = N_current_scatter_groups / N_macro_gaussian_pairs if N_macro_gaussian_pairs > 0 else 0

    print(f"N_current_scatter_groups = {N_current_scatter_groups}")
    print(f"R_crosswarp = {R_crosswarp:.4f}")
    print(f"R_total = {R_total:.4f}")
    print(f"Warp coverage: mean={np.mean(n_warps_per_isect):.2f} "
          f"p50={np.percentile(n_warps_per_isect,50):.0f} "
          f"p95={np.percentile(n_warps_per_isect,95):.0f} "
          f"max={np.max(n_warps_per_isect)}")

    # Warp coverage histogram
    warp_hist = {}
    for v in n_warps_per_isect:
        warp_hist[int(v)] = warp_hist.get(int(v), 0) + 1
    print(f"Warp coverage histogram: {dict(sorted(warp_hist.items()))}")

    # ============================================================
    # SECTION 6: Combined structural reduction opportunity
    # ============================================================
    print("\n===== Section 6: Combined structural reduction =====")
    print(f"R_macro = {R_macro:.4f}")
    print(f"R_crosswarp = {R_crosswarp:.4f}")
    print(f"R_total = {R_total:.4f}")
    print(f"(R_total is gradient-scatter compression, NOT expected speedup)")

    # ============================================================
    # SECTION 8: Macro-hierarchy transformation cost estimate
    # ============================================================
    print("\n===== Section 8: Macro-hierarchy transformation cost =====")
    # Compact batch state: P_b (prefix transmittance) + lambda_b (reverse scalar adjoint)
    # = 2 * FP32 = 8 bytes per active pixel-batch
    # Plus uint16 last_local_id = 2 bytes
    # Total = 10 bytes per active pixel-batch

    # ============================================================
    # SECTION 9: Batch count / state expansion
    # ============================================================
    print("\n===== Section 9: Batch state memory =====")
    # For each active fine-tile pixel, it participates in all batches of its macro tile
    # N_active_pixel_batches = sum over macro tiles of (n_batches * n_pixels_in_macro)
    # Vectorized: for each pixel, find its macro tile, multiply by n_batches for that macro

    width, height = fwd["width"], fwd["height"]
    # Pixel -> tile -> macro mapping (vectorized)
    px_tile_x = np.arange(width) // TILE_SIZE
    px_tile_y = np.arange(height) // TILE_SIZE
    # 2D grid of macro_ids
    px_macro_x = px_tile_x // FUSED_MACRO_TILE_WIDTH
    px_macro_y = px_tile_y // FUSED_MACRO_TILE_HEIGHT
    # macro_id per pixel
    px_macro_ids = px_macro_y[None, :] * macro_width + px_macro_x[:, None]  # [height, width]
    px_macro_flat = px_macro_ids.reshape(-1).astype(np.int32)

    # n_batches per macro tile
    n_batches_full = np.ceil(n_unique_gauss_per_macro / FUSED_GAUSS_BATCH_SIZE).astype(np.int32)
    n_batches_per_pixel = n_batches_full[px_macro_flat]
    N_active_pixel_batches = int(np.sum(n_batches_per_pixel))

    prefix_adjoint_bytes = N_active_pixel_batches * 8
    last_local_id_bytes = N_active_pixel_batches * 2
    total_batch_state_bytes = prefix_adjoint_bytes + last_local_id_bytes

    print(f"N_active_pixel_batches = {N_active_pixel_batches}")
    print(f"prefix/adjoint state = {prefix_adjoint_bytes / 1e6:.2f} MB")
    print(f"last_local_id = {last_local_id_bytes / 1e6:.2f} MB")
    print(f"Total batch state = {total_batch_state_bytes / 1e6:.2f} MB")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "scene": scene, "camera_idx": cam_idx,
        "width": width, "height": height,
        "N_total": fwd["N_total"], "N_visible": N_visible,
        "tile_width": tile_width, "tile_height": tile_height, "n_tiles": n_tiles,
        "macro_width": macro_width, "macro_height": macro_height, "n_macro_tiles": n_macro_tiles,
        "n_macro_tiles_active": n_active_macros,
        "section1_macro_pair_oracle": {
            "N_tile_gaussian_pairs": int(n_isects),
            "N_macro_gaussian_pairs": int(N_macro_gaussian_pairs),
            "R_macro": float(R_macro),
        },
        "section2_fine_tile_reuse": {
            "mean": mean_ft, "p50": p50_ft, "p75": p75_ft, "p90": p90_ft,
            "p95": p95_ft, "p99": p99_ft, "max": max_ft,
            "histogram": dict(zip([str(b) for b in hist_bins], hist_counts)),
        },
        "section3_macro_workload": {
            "N_unique_gaussians": {
                "mean": float(np.mean(macro_unique_gauss)), "p50": float(np.percentile(macro_unique_gauss, 50)),
                "p90": float(np.percentile(macro_unique_gauss, 90)), "p95": float(np.percentile(macro_unique_gauss, 95)),
                "p99": float(np.percentile(macro_unique_gauss, 99)), "max": int(np.max(macro_unique_gauss)),
            },
            "N_fine_tile_gaussian_pairs": {
                "mean": float(np.mean(macro_pairs)), "p50": float(np.percentile(macro_pairs, 50)),
                "p90": float(np.percentile(macro_pairs, 90)), "p95": float(np.percentile(macro_pairs, 95)),
                "p99": float(np.percentile(macro_pairs, 99)), "max": int(np.max(macro_pairs)),
            },
            "N_active_fine_tiles": {
                "mean": float(np.mean(macro_active_tiles)), "p50": float(np.percentile(macro_active_tiles, 50)),
                "p90": float(np.percentile(macro_active_tiles, 90)), "p95": float(np.percentile(macro_active_tiles, 95)),
                "p99": float(np.percentile(macro_active_tiles, 99)), "max": int(np.max(macro_active_tiles)),
            },
            "N_1024_batches": {
                "mean": float(np.mean(n_batches_per_macro)), "p50": float(np.percentile(n_batches_per_macro, 50)),
                "p90": float(np.percentile(n_batches_per_macro, 90)), "p95": float(np.percentile(n_batches_per_macro, 95)),
                "p99": float(np.percentile(n_batches_per_macro, 99)), "max": int(np.max(n_batches_per_macro)),
            },
            "batch_hist": {str(k): v for k, v in sorted(batch_hist.items())},
            "n_active_macros": n_active_macros,
        },
        "section4_mini_batch_feasibility": {
            "grad_state_bytes": grad_state_bytes,
            "metadata_bytes": metadata_bytes,
            "total_smem_estimate": total_smem_estimate,
            "full_batch_smem": full_batch_smem,
            "est_blocks_per_sm": est_blocks_per_sm,
            "feasibility": feasibility,
        },
        "section5_scatter_multiplicity": {
            "N_current_scatter_groups": int(N_current_scatter_groups),
            "R_crosswarp": float(R_crosswarp),
            "warp_coverage_mean": float(np.mean(n_warps_per_isect)),
            "warp_coverage_p50": float(np.percentile(n_warps_per_isect, 50)),
            "warp_coverage_p95": float(np.percentile(n_warps_per_isect, 95)),
            "warp_coverage_max": int(np.max(n_warps_per_isect)),
            "warp_coverage_histogram": {str(k): v for k, v in sorted(warp_hist.items())},
        },
        "section6_combined_reduction": {
            "R_macro": float(R_macro),
            "R_crosswarp": float(R_crosswarp),
            "R_total": float(R_total),
        },
        "section9_batch_state_memory": {
            "N_active_pixel_batches": int(N_active_pixel_batches),
            "prefix_adjoint_MB": prefix_adjoint_bytes / 1e6,
            "last_local_id_MB": last_local_id_bytes / 1e6,
            "total_MB": total_batch_state_bytes / 1e6,
        },
    }

    # Save JSON
    fname = f"{scene}_cam{cam_idx}_structural.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(results, f, indent=2)

    # Save histogram CSV
    hname = f"{scene}_cam{cam_idx}_histogram.csv"
    with open(os.path.join(out_dir, hname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fine_tile_count_bucket", "count"])
        for b, c in zip(hist_bins, hist_counts):
            w.writerow([b, c])

    # Save macro workload CSV
    wname = f"{scene}_cam{cam_idx}_macro_workload.csv"
    with open(os.path.join(out_dir, wname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["macro_id", "n_unique_gaussians", "n_fine_tile_gaussian_pairs", "n_active_fine_tiles", "n_1024_batches"])
        for mid in range(n_macro_tiles):
            w.writerow([mid, int(n_unique_gauss_per_macro[mid]), int(n_pairs_per_macro[mid]),
                       int(n_active_tiles_per_macro[mid]), int(n_batches_full[mid])])

    print(f"\n[done] Results saved to {out_dir}/{fname}")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-long-side", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    print(f"[H2-BWD-0] Structural analysis: {args.scene}/cam{args.cam_idx}")
    fwd = run_forward_get_isects(args, device)
    results = compute_structural_stats(fwd, args.scene, args.cam_idx, args.out_dir)

if __name__ == "__main__":
    main()
