#!/usr/bin/env python3
"""
R6-A: Warp/block duplicate analysis for atomic reduction potential.

**SUPERSEDED by r6_a_direct_atomic.py** — this script uses a footprint-based
ESTIMATE of within-tile warp multiplicity. The direct pixel-level simulation
in r6_a_direct_atomic.py provides more accurate measurements.

KNOWN DIMENSIONAL ISSUE (corrected in r6_a_direct_atomic.py):
  - `n_warps_per_gaussian` is labeled "warps per Gaussian" but is actually
    "within-tile warps per Gaussian" (per-tile, not total across all tiles)
  - The correct total warps per Gaussian = tiles_per_gaussian × within_tile_warps
  - The original "cross-tile vs cross-warp" decomposition was nonsensical
  - See r6_a_direct_atomic.py for corrected definitions and direct measurement

gsplat 1.5.3's backward kernel ALREADY does warp-level aggregation:
- All 32 lanes of a warp process the SAME Gaussian t
- warpSum reduces across lanes
- Only warp leader (thread_rank==0) issues gpuAtomicAdd

So the relevant duplicate is NOT within-warp (already aggregated) but:
1. CROSS-WARP (within tile): how many of the 8 warps in a tile write to the same Gaussian?
2. CROSS-TILE: how many tiles contain the same Gaussian?

This script computes (ESTIMATED, not direct):
- Per-Gaussian tile intersection count (cross-tile duplicates)
- Per-tile warp coverage estimate (cross-warp duplicates) — FOOTPRINT-BASED, overestimates
- R_atomic_potential = total_warp_gaussian_pairs / total_tile_gaussian_pairs

CORRECTED definitions:
  tiles_per_gaussian (T/G): number of tiles each visible Gaussian intersects
  within_tile_warps (W/tile/G): warps per Gaussian within each tile (THIS script estimates it)
  R_atomic = W/tile/G (weighted): block-aggregation reduction factor
  total_warps_per_gaussian = T/G × R_atomic: total warp-Gaussian pairs per Gaussian

Usage (on mx):
  CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/r6_4_warp_duplicate.py \
    --scene room --ckpt /mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_5000.pt \
    --output /mnt/storage_pool/liaoyuanjun/r6_profiling/r6_4_room_5k.json
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, argparse, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

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


def analyze_intersection_structure(meta, tile_size=16, image_width=None, image_height=None):
    """Analyze cross-tile and cross-warp duplicate structure from forward metadata.

    The intersection list (flatten_ids + isect_offsets) tells us which Gaussians
    intersect each tile. From this we compute:
    - Per-Gaussian tile count (cross-tile duplicates)
    - Per-tile intersection count (workload per tile)
    - Total intersections = sum of per-tile counts

    For cross-warp duplicates, we estimate each Gaussian's warp coverage within
    each tile using the Gaussian's 2D footprint (from means2d + conics).
    """
    isect_offsets = meta["isect_offsets"]  # [tile_height, tile_width] or [1, th, tw]
    flatten_ids = meta["flatten_ids"]  # [n_isects]
    means2d = meta["means2d"][0]  # [N, 2]
    conics = meta["conics"][0]  # [N, 3]
    radii = meta["radii"][0]  # [N, 2]

    if isect_offsets.dim() == 3:
        isect_offsets = isect_offsets[0]  # [th, tw]

    th, tw = isect_offsets.shape
    n_isects = flatten_ids.shape[0]

    # === Cross-tile analysis ===
    # For each Gaussian, count how many tiles it intersects
    # flatten_ids contains Gaussian IDs in tile-sorted order
    gaussian_tile_counts = torch.zeros(means2d.shape[0], dtype=torch.int32, device="cuda")
    gaussian_tile_counts.scatter_add_(0, flatten_ids.long(),
                                       torch.ones_like(flatten_ids, dtype=torch.int32))

    # Only count visible Gaussians
    visible = (radii > 0).any(dim=-1)
    vis_gtc = gaussian_tile_counts[visible]

    # === Per-tile intersection counts ===
    # tile_intersections[t] = isect_offsets[t+1] - isect_offsets[t]
    # Need to handle the last tile
    offsets_flat = isect_offsets.flatten()
    tile_intersects = torch.zeros(th * tw, dtype=torch.int32, device="cuda")
    tile_intersects[:-1] = offsets_flat[1:] - offsets_flat[:-1]
    tile_intersects[-1] = n_isects - offsets_flat[-1]

    # === Cross-warp estimate ===
    # Each tile is tile_size × tile_size = 16×16 = 256 threads = 8 warps.
    # Each warp covers 2 rows × 16 cols (threadIdx layout: ty*16+tx, warp = floor(idx/32)).
    # A Gaussian with 2D footprint radius r (in pixels) spans approximately
    # ceil(2r / 2) = ceil(r) warp-rows within the tile.
    # But the Gaussian may not span the full tile width.
    #
    # We estimate warp coverage per Gaussian per tile as:
    #   n_warps ≈ max(1, ceil(footprint_height / warp_height))
    # where warp_height = 2 (each warp covers 2 pixel rows for tile_size=16).
    #
    # The footprint height can be estimated from the conic:
    # For a 2D Gaussian with conic [a, b, c], the "radius" at alpha threshold
    # is approximately sqrt(-log(threshold) / min_eigenvalue).
    # We use a simpler proxy: the 2D bounding box from radii.

    # Estimate footprint radius from radii (which are the 2D radii in pixels)
    # radii is [N, 2] — the bounding box semi-axes
    footprint_height = radii[visible, 1].float()  # [N_vis]
    # Clamp to tile_size (a Gaussian can't span more than the tile)
    footprint_height_tile = torch.clamp(footprint_height, min=1.0, max=float(tile_size))

    # Warp coverage: each warp covers 2 rows, so n_warps = ceil(height / 2)
    warp_height = 2.0  # for tile_size=16, each warp covers 2 rows
    n_warps_per_gaussian = torch.ceil(footprint_height_tile / warp_height).int()
    n_warps_per_gaussian = torch.clamp(n_warps_per_gaussian, min=1, max=8)

    # Weight by tile count: total warp-Gaussian pairs
    vis_tile_counts = vis_gtc.float()
    total_warp_gaussian_pairs = (n_warps_per_gaussian.float() * vis_tile_counts).sum().item()
    total_tile_gaussian_pairs = vis_tile_counts.sum().item()  # = n_isects (for visible)

    # R_atomic_potential = total_warp_gaussian_pairs / total_tile_gaussian_pairs
    # This is the average number of warps per (tile, Gaussian) pair
    # If >1, block-level aggregation could reduce atomics by this factor
    R_atomic = total_warp_gaussian_pairs / max(total_tile_gaussian_pairs, 1)

    # === Statistics ===
    stats = {
        "n_isects": int(n_isects),
        "n_tiles": int(th * tw),
        "n_tiles_with_isects": int((tile_intersects > 0).sum().item()),
        "tile_intersects_mean": float(tile_intersects.float().mean().item()),
        "tile_intersects_median": float(tile_intersects.float().median().item()),
        "tile_intersects_p95": float(tile_intersects.float().quantile(0.95).item()),
        "tile_intersects_max": int(tile_intersects.max().item()),
        "gaussian_tile_counts_mean": float(vis_gtc.float().mean().item()),
        "gaussian_tile_counts_median": float(vis_gtc.float().median().item()),
        "gaussian_tile_counts_p10": float(vis_gtc.float().quantile(0.10).item()),
        "gaussian_tile_counts_p90": float(vis_gtc.float().quantile(0.90).item()),
        "gaussian_tile_counts_max": int(vis_gtc.max().item()),
        "gaussians_in_1_tile": int((vis_gtc == 1).sum().item()),
        "gaussians_in_2to4_tiles": int(((vis_gtc >= 2) & (vis_gtc <= 4)).sum().item()),
        "gaussians_in_5plus_tiles": int((vis_gtc >= 5).sum().item()),
        "total_tile_gaussian_pairs": int(total_tile_gaussian_pairs),
        "total_warp_gaussian_pairs_est": int(total_warp_gaussian_pairs),
        "R_atomic_potential_est": R_atomic,
        "n_warps_per_gaussian_mean": float(n_warps_per_gaussian.float().mean().item()),
        "n_warps_per_gaussian_median": float(n_warps_per_gaussian.float().median().item()),
        "n_warps_per_gaussian_p90": float(n_warps_per_gaussian.float().quantile(0.90).item()),
        "footprint_height_mean": float(footprint_height.mean().item()),
        "footprint_height_median": float(footprint_height.median().item()),
        "footprint_height_p90": float(footprint_height.quantile(0.90).item()),
    }

    # Hypothetical scenarios
    stats["atomic_count_half"] = total_warp_gaussian_pairs / 2
    stats["atomic_count_quarter"] = total_warp_gaussian_pairs / 4
    stats["atomic_count_eighth"] = total_warp_gaussian_pairs / 8
    stats["ideal_block_aggregated"] = total_tile_gaussian_pairs

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cameras", type=int, default=10,
                        help="Number of cameras to profile")
    args = parser.parse_args()

    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = str(REPO_ROOT)
    config.resolution = args.resolution
    config.seed = args.seed

    print(f"=== R6-A Warp Duplicate Analysis: {args.scene} ===")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Cameras: {args.n_cameras}")

    dataset = GTDataset(
        scene=config.scene, repo_root=config.repo_root,
        resolution=config.resolution, device="cuda", background="black",
    )
    print(f"  {len(dataset)} cameras loaded")

    model = load_model_from_ckpt(args.ckpt, config, config.repo_root)
    N_total = model._xyz.shape[0]
    print(f"  Model: N={N_total}, SH degree={model.active_sh_degree}")

    # Profile multiple cameras
    n_cam = min(args.n_cameras, len(dataset))
    all_stats = []
    for cam_idx in range(n_cam):
        cam, gt_image = dataset.get_item(cam_idx)
        with torch.no_grad():
            image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree)

        stats = analyze_intersection_structure(meta, tile_size=16)
        stats["camera_idx"] = cam_idx
        all_stats.append(stats)

        print(f"  Cam {cam_idx}: n_isects={stats['n_isects']}, "
              f"R_atomic_est={stats['R_atomic_potential_est']:.2f}, "
              f"tiles/Gauss={stats['gaussian_tile_counts_mean']:.1f}, "
              f"warps/Gauss={stats['n_warps_per_gaussian_mean']:.1f}")

    # Aggregate
    R_atomic_vals = [s["R_atomic_potential_est"] for s in all_stats]
    n_isects_vals = [s["n_isects"] for s in all_stats]
    tiles_per_gauss = [s["gaussian_tile_counts_mean"] for s in all_stats]
    warps_per_gauss = [s["n_warps_per_gaussian_mean"] for s in all_stats]

    result = {
        "scene": args.scene,
        "checkpoint": args.ckpt,
        "N_total": N_total,
        "n_cameras_profiled": n_cam,
        "per_camera": all_stats,
        "aggregate": {
            "R_atomic_mean": float(np.mean(R_atomic_vals)),
            "R_atomic_median": float(np.median(R_atomic_vals)),
            "R_atomic_std": float(np.std(R_atomic_vals)),
            "n_isects_mean": float(np.mean(n_isects_vals)),
            "tiles_per_gauss_mean": float(np.mean(tiles_per_gauss)),
            "within_tile_warps_per_gauss_mean": float(np.mean(warps_per_gauss)),  # CORRECTED LABEL: was warps_per_gauss_mean
            "total_warps_per_gauss_est": float(np.mean(tiles_per_gauss)) * float(np.mean(R_atomic_vals)),  # ADDED: T/G × R_atomic
        },
    }

    print(f"\n=== Aggregate ({n_cam} cameras) ===")
    print(f"  R_atomic_potential: mean={result['aggregate']['R_atomic_mean']:.2f}, "
          f"median={result['aggregate']['R_atomic_median']:.2f}")
    print(f"  Tiles per Gaussian: {result['aggregate']['tiles_per_gauss_mean']:.2f}")
    print(f"  Warps per Gaussian: {result['aggregate']['warps_per_gauss_mean']:.2f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
