#!/usr/bin/env python3
"""
R6-A DIRECT atomic instrumentation — replaces the footprint-based estimate.

The original r6_4_warp_duplicate.py estimated within-tile warp multiplicity
from bounding-box footprints. This script provides DIRECT measurement:

1. Cross-tile duplication (DIRECT):
   - n_isects = total (tile, Gaussian) pairs from forward metadata
   - n_visible = count of Gaussians with radii > 0 (unique rasterizer targets)
   - cross_tile_dup = n_isects / n_visible

2. Within-tile warp multiplicity (DIRECT, pixel-level simulation on subset):
   - For a random subset of tiles, evaluate each Gaussian's alpha at every pixel
   - Map pixels to warps (warp_id = pixel_idx // 32)
   - Count warp-Gaussian pairs where any pixel in the warp has alpha > threshold
   - within_tile_warps = total_warp_gaussian_pairs / total_tile_gaussian_pairs (on subset)

3. R_atomic = within_tile_warps (the block-aggregation reduction factor)

4. Block oracle: atomic_count_block = n_isects (1 per tile-Gaussian pair)
   Baseline: atomic_count_baseline = within_tile_warps × n_isects
   R_atomic = baseline / oracle = within_tile_warps

Dimensional definitions (CORRECTED):
  tiles_per_gaussian (T/G): number of tiles each visible Gaussian intersects
  within_tile_warps (W/tile/G): warps per Gaussian within each tile (from pixel sim)
  total_warps_per_gaussian = T/G × W/tile/G: total warp-Gaussian pairs per Gaussian
  R_atomic = W/tile/G (weighted): block-aggregation reduction factor

Usage (on mx):
  CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/r6_a_direct_atomic.py \
    --scene room --ckpt /mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_5000.pt \
    --output /mnt/storage_pool/liaoyuanjun/r6_profiling/r6_a_direct_room_5k.json \
    --n-cameras 5 --n-tiles-sample 200
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, argparse, math, random
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization
from dataset import GTDataset
from trainer import SepSSIM, render_with_meta


def load_model_from_ckpt(ckpt_path, config, repo_root):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    model.active_sh_degree = ckpt["active_sh_degree"]
    return model


def direct_atomic_analysis(meta, tile_size=16, alpha_threshold=0.01, n_tiles_sample=200, device="cuda"):
    """
    Direct measurement of atomic duplication structure.

    Returns:
      cross_tile_dup: n_isects / n_visible (DIRECT)
      within_tile_warps: pixel-level simulation (DIRECT on subset)
      R_atomic: within_tile_warps (block-aggregation reduction factor)
    """
    isect_offsets = meta["isect_offsets"]
    flatten_ids = meta["flatten_ids"]
    means2d = meta["means2d"][0]      # [N, 2]
    conics = meta["conics"][0]        # [N, 3]
    radii = meta["radii"][0]          # [N, 2]
    opacities = meta["opacities"][0]  # [N] or [1, N]

    if isect_offsets.dim() == 3:
        isect_offsets = isect_offsets[0]
    if opacities.dim() == 2:
        opacities = opacities[0]

    th, tw = isect_offsets.shape
    n_isects = flatten_ids.shape[0]
    N = means2d.shape[0]

    # === DIRECT: Cross-tile duplication ===
    # Count tiles per visible Gaussian
    gaussian_tile_counts = torch.zeros(N, dtype=torch.int32, device=device)
    gaussian_tile_counts.scatter_add_(0, flatten_ids.long(),
                                       torch.ones_like(flatten_ids, dtype=torch.int32))

    visible = (radii > 0).any(dim=-1)
    n_visible = int(visible.sum().item())
    vis_gtc = gaussian_tile_counts[visible]  # tile count per visible Gaussian

    # Cross-tile duplication = total intersections / unique visible Gaussians
    # This is the AVERAGE number of tiles per visible Gaussian (weighted by intersections)
    cross_tile_dup = n_isects / max(n_visible, 1)

    # Per-tile intersection counts
    offsets_flat = isect_offsets.flatten()
    tile_intersects = torch.zeros(th * tw, dtype=torch.int32, device=device)
    tile_intersects[:-1] = offsets_flat[1:] - offsets_flat[:-1]
    tile_intersects[-1] = n_isects - offsets_flat[-1]

    # Tiles with intersections
    active_tiles = torch.where(tile_intersects > 0)[0]
    n_active_tiles = int(active_tiles.shape[0])

    # === DIRECT: Within-tile warp multiplicity (pixel-level simulation) ===
    # Sample a subset of active tiles
    n_sample = min(n_tiles_sample, n_active_tiles)
    if n_sample < n_active_tiles:
        perm = torch.randperm(n_active_tiles, device=device)[:n_sample]
        sampled_tile_ids = active_tiles[perm]
    else:
        sampled_tile_ids = active_tiles

    tile_y = (sampled_tile_ids // tw).long()  # [n_sample]
    tile_x = (sampled_tile_ids % tw).long()   # [n_sample]

    total_warp_gaussian_pairs_sample = 0
    total_tile_gaussian_pairs_sample = 0
    per_tile_results = []

    for i in range(n_sample):
        ty, tx = int(tile_y[i]), int(tile_x[i])
        tile_start = int(isect_offsets[ty, tx].item())
        if ty == th - 1 and tx == tw - 1:
            tile_end = n_isects
        elif tx == tw - 1:
            tile_end = int(isect_offsets[ty + 1, 0].item())
        else:
            tile_end = int(isect_offsets[ty, tx + 1].item())

        if tile_end <= tile_start:
            continue

        gids = flatten_ids[tile_start:tile_end].long()  # [K]
        K = gids.shape[0]
        if K == 0:
            continue

        # Pixel positions in this tile (tile_size × tile_size)
        # Pixel (py, px) in tile → global pixel (ty*tile_size + py, tx*tile_size + px)
        py = torch.arange(tile_size, device=device)
        px = torch.arange(tile_size, device=device)
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")  # [ts, ts]
        pixel_y = grid_y.flatten() + ty * tile_size  # [ts²]
        pixel_x = grid_x.flatten() + tx * tile_size  # [ts²]
        pixel_pos = torch.stack([pixel_x, pixel_y], dim=-1).float()  # [ts², 2]

        # For each Gaussian in this tile, evaluate alpha at each pixel
        g_means = means2d[gids]      # [K, 2]
        g_conics = conics[gids]      # [K, 3]
        g_opac = opacities[gids]     # [K]

        # delta = pixel_pos[None,:,:] - g_means[:,None,:]  → [K, ts², 2]
        delta = pixel_pos.unsqueeze(0) - g_means.unsqueeze(1)  # [K, ts², 2]

        # alpha = opacity * exp(-0.5 * delta^T * conic * delta)
        # conic = [a, b, c], delta = [dx, dy]
        # delta^T * conic * delta = a*dx² + 2*b*dx*dy + c*dy²
        a = g_conics[:, 0:1]  # [K, 1]
        b = g_conics[:, 1:2]
        c = g_conics[:, 2:3]
        dx = delta[..., 0]    # [K, ts²]
        dy = delta[..., 1]
        mahalanobis = a * dx * dx + 2 * b * dx * dy + c * dy * dy  # [K, ts²]
        mahalanobis = torch.clamp(mahalanobis, min=0)
        alpha = g_opac.unsqueeze(1) * torch.exp(-0.5 * mahalanobis)  # [K, ts²]

        # Which pixels are covered (alpha > threshold)?
        covered = alpha > alpha_threshold  # [K, ts²]

        # Map pixels to warps: warp_id = pixel_idx // 32
        n_pixels = tile_size * tile_size
        warp_ids = torch.arange(n_pixels, device=device) // 32  # [ts²]

        # For each Gaussian, count unique warps with any covered pixel
        # covered[k] = [ts²] bool, warp_ids = [ts²]
        # For each warp w, check if any pixel in warp w is covered
        n_warps_per_tile = n_pixels // 32  # 8 for tile_size=16
        # Vectorized: [K, n_warps] where entry[k,w] = any(covered[k, warp_ids==w])
        warp_counts = torch.zeros(K, dtype=torch.int32, device=device)
        for w in range(n_warps_per_tile):
            warp_mask = warp_ids == w  # [ts²]
            in_warp = covered[:, warp_mask].any(dim=1)  # [K]
            warp_counts += in_warp.int()

        # Clamp to [1, n_warps_per_tile] (at least 1 warp since Gaussian is in the tile)
        warp_counts = warp_counts.clamp(min=1, max=n_warps_per_tile)

        total_warp_gaussian_pairs_sample += int(warp_counts.sum().item())
        total_tile_gaussian_pairs_sample += K

        per_tile_results.append({
            "tile_id": int(sampled_tile_ids[i]),
            "n_gaussians": K,
            "warp_gaussian_pairs": int(warp_counts.sum().item()),
            "mean_warps_per_gaussian": float(warp_counts.float().mean().item()),
        })

    # Within-tile warp multiplicity (from pixel simulation)
    within_tile_warps = total_warp_gaussian_pairs_sample / max(total_tile_gaussian_pairs_sample, 1)

    # R_atomic = within_tile_warps (the block-aggregation reduction factor)
    R_atomic = within_tile_warps

    # Total warp-Gaussian pairs (extrapolated from sample to full image)
    # Using the sample's within_tile_warps × full n_isects
    total_warp_gaussian_pairs_full = R_atomic * n_isects

    # Block oracle: 1 atomic per (tile, Gaussian) = n_isects
    atomic_count_baseline = total_warp_gaussian_pairs_full
    atomic_count_block_oracle = n_isects
    reduction_factor = atomic_count_baseline / max(atomic_count_block_oracle, 1)

    return {
        "n_isects": n_isects,
        "n_visible": n_visible,
        "n_active_tiles": n_active_tiles,
        "n_total_tiles": int(th * tw),
        "cross_tile_dup_direct": cross_tile_dup,  # n_isects / n_visible
        "tiles_per_gaussian_mean": float(vis_gtc.float().mean().item()),
        "tiles_per_gaussian_median": float(vis_gtc.float().median().item()),
        "tiles_per_gaussian_p90": float(vis_gtc.float().quantile(0.90).item()),
        "within_tile_warps_direct": within_tile_warps,  # from pixel simulation
        "R_atomic": R_atomic,
        "total_warp_gaussian_pairs_sample": total_warp_gaussian_pairs_sample,
        "total_tile_gaussian_pairs_sample": total_tile_gaussian_pairs_sample,
        "total_warp_gaussian_pairs_full_est": total_warp_gaussian_pairs_full,
        "atomic_count_baseline": atomic_count_baseline,
        "atomic_count_block_oracle": atomic_count_block_oracle,
        "reduction_factor": reduction_factor,
        "total_warps_per_gaussian": float(vis_gtc.float().mean().item()) * R_atomic,
        "n_tiles_sampled": n_sample,
        "per_tile_sample": per_tile_results[:20],  # first 20 for inspection
        "alpha_threshold": alpha_threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cameras", type=int, default=5)
    parser.add_argument("--n-tiles-sample", type=int, default=200,
                        help="Number of tiles to sample for pixel-level simulation")
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = str(REPO_ROOT)
    config.resolution = args.resolution
    config.seed = args.seed

    print(f"=== R6-A DIRECT Atomic Instrumentation: {args.scene} ===")
    dataset = GTDataset(scene=config.scene, repo_root=config.repo_root,
                        resolution=config.resolution, device="cuda", background="black")
    model = load_model_from_ckpt(args.ckpt, config, config.repo_root)
    N_total = model._xyz.shape[0]
    print(f"  N={N_total}, SH={model.active_sh_degree}")

    all_stats = []
    for cam_idx in range(min(args.n_cameras, len(dataset))):
        cam, gt_image = dataset.get_item(cam_idx)
        with torch.no_grad():
            image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree)

        stats = direct_atomic_analysis(meta, tile_size=16,
                                        alpha_threshold=args.alpha_threshold,
                                        n_tiles_sample=args.n_tiles_sample)
        stats["camera_idx"] = cam_idx
        all_stats.append(stats)

        print(f"  Cam {cam_idx}: n_isects={stats['n_isects']}, "
              f"n_visible={stats['n_visible']}, "
              f"cross_tile_dup={stats['cross_tile_dup_direct']:.2f}, "
              f"within_tile_warps={stats['within_tile_warps_direct']:.2f}, "
              f"R_atomic={stats['R_atomic']:.2f}, "
              f"reduction={stats['reduction_factor']:.2f}")

    # Aggregate
    cross_tile_dups = [s["cross_tile_dup_direct"] for s in all_stats]
    within_tile_warps = [s["within_tile_warps_direct"] for s in all_stats]
    R_atomics = [s["R_atomic"] for s in all_stats]
    reductions = [s["reduction_factor"] for s in all_stats]
    n_isects_vals = [s["n_isects"] for s in all_stats]
    n_visible_vals = [s["n_visible"] for s in all_stats]

    result = {
        "scene": args.scene,
        "checkpoint": args.ckpt,
        "N_total": N_total,
        "n_cameras": len(all_stats),
        "method": "direct_pixel_simulation",
        "alpha_threshold": args.alpha_threshold,
        "n_tiles_sampled": args.n_tiles_sample,
        "per_camera": all_stats,
        "aggregate": {
            "cross_tile_dup_mean": float(np.mean(cross_tile_dups)),
            "cross_tile_dup_std": float(np.std(cross_tile_dups)),
            "within_tile_warps_mean": float(np.mean(within_tile_warps)),
            "within_tile_warps_std": float(np.std(within_tile_warps)),
            "R_atomic_mean": float(np.mean(R_atomics)),
            "R_atomic_std": float(np.std(R_atomics)),
            "reduction_factor_mean": float(np.mean(reductions)),
            "reduction_factor_std": float(np.std(reductions)),
            "n_isects_mean": float(np.mean(n_isects_vals)),
            "n_visible_mean": float(np.mean(n_visible_vals)),
        },
    }

    print(f"\n=== AGGREGATE ({len(all_stats)} cameras) ===")
    print(f"  Cross-tile duplication: {result['aggregate']['cross_tile_dup_mean']:.2f} "
          f"± {result['aggregate']['cross_tile_dup_std']:.2f}")
    print(f"  Within-tile warps: {result['aggregate']['within_tile_warps_mean']:.2f} "
          f"± {result['aggregate']['within_tile_warps_std']:.2f}")
    print(f"  R_atomic (block reduction factor): {result['aggregate']['R_atomic_mean']:.2f} "
          f"± {result['aggregate']['R_atomic_std']:.2f}")
    print(f"  Reduction factor: {result['aggregate']['reduction_factor_mean']:.2f}x")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
