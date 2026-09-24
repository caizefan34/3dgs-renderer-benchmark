#!/usr/bin/env python3
"""
R3-0: W_it faithful replay validation.

Compares the vectorized cumprod GPU replay against an independent
scalar/reference Python replay for >= 128 random (tile, Gaussian) pairs
from one real Room camera.

Invariants:
  0 <= W_unclamped <= W_color <= 256 (tile16)
  W_COLOR_MISMATCHES = 0
  W_UNCLAMPED_MISMATCHES = 0
"""

import os, sys, time, math, random, json, argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline", "reference_v1")
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat.cuda._wrapper import (
    fully_fused_projection, spherical_harmonics,
    isect_tiles, isect_offset_encode, rasterize_to_pixels,
)
from benchmark_framework import load_cameras_from_json, resize_cameras
from pathlib import Path

torch.set_printoptions(linewidth=200, sci_mode=False)


# ============================================================
# Faithful vectorized GPU replay
# ============================================================
def _get_tile_pixels(tile_x, tile_y, tile_size, H, W, device,
                     n_subsample=None):
    """Returns (px_flat, py_flat) for tile pixels.
    If n_subsample is set, returns a random subset (with fixed seed 42)."""
    px_min = tile_x
    px_max = min(tile_x + tile_size, W)
    py_min = tile_y
    py_max = min(tile_y + tile_size, H)

    if px_min >= px_max or py_min >= py_max:
        return None, None

    px_vals = torch.arange(px_min, px_max, device=device, dtype=torch.float32) + 0.5
    py_vals = torch.arange(py_min, py_max, device=device, dtype=torch.float32) + 0.5
    px_grid, py_grid = torch.meshgrid(px_vals, py_vals, indexing='xy')
    px_flat = px_grid.reshape(-1)
    py_flat = py_grid.reshape(-1)
    P = len(px_flat)

    if n_subsample is not None and n_subsample < P:
        # Fixed seed 42 for reproducibility
        rng = torch.manual_seed(42)
        perm = torch.randperm(P, device=device)[:n_subsample]
        px_flat = px_flat[perm]
        py_flat = py_flat[perm]

    return px_flat, py_flat


def compute_w_gpu(g_indices, means2d_0, conics_0, opacities,
                  px_flat, py_flat):
    """
    GPU-vectorized cumprod replay.
    px_flat, py_flat: specific pixel coordinates to evaluate.
    Returns (w_color_dict, w_unclamped_dict).
    """
    G = len(g_indices)
    if G == 0 or px_flat is None:
        return {}, {}

    device = opacities.device
    gi_tensor = g_indices.to(device) if not g_indices.is_cuda else g_indices.long()

    mx = means2d_0[gi_tensor, 0]
    my = means2d_0[gi_tensor, 1]
    xx = conics_0[gi_tensor, 0]
    xy = conics_0[gi_tensor, 1]
    yy = conics_0[gi_tensor, 2]
    opac = opacities[gi_tensor]

    P = len(px_flat)

    dx = px_flat[:, None] - mx[None, :]
    dy = py_flat[:, None] - my[None, :]
    sigma = (0.5 * (xx[None, :] * dx * dx + yy[None, :] * dy * dy) +
             xy[None, :] * dx * dy)

    exp_minus_sigma = torch.exp(-sigma)
    alpha_raw = opac[None, :] * exp_minus_sigma
    alpha = torch.minimum(alpha_raw, torch.full_like(alpha_raw, 0.999))

    skip = (sigma < 0.0) | (alpha < (1.0 / 255.0))
    ra = torch.where(skip, torch.ones_like(alpha), 1.0 - alpha)
    cp = torch.cumprod(ra, dim=1)
    contributes = (~skip) & (cp > 1e-4)

    w_color = contributes.sum(dim=0).to(torch.int32)
    unclamped = contributes & (alpha_raw <= 0.999)
    w_unclamped = unclamped.sum(dim=0).to(torch.int32)

    gi_list = [int(gi) for gi in g_indices]
    wc_dict = {gi: int(w_color[i].item()) for i, gi in enumerate(gi_list)}
    wu_dict = {gi: int(w_unclamped[i].item()) for i, gi in enumerate(gi_list)}
    return wc_dict, wu_dict


# ============================================================
# Independent scalar reference replay
# ============================================================
def compute_w_scalar(g_indices, means2d_0, conics_0, opacities,
                     px_flat, py_flat):
    """
    Independent scalar Python reference: pure Python float arithmetic.
    px_flat, py_flat: pixel coordinates to evaluate (MUST match GPU).

    Faithfully replicates gsplat rasterize_to_pixels_fwd semantics:
      - sigma = 0.5*(xx*dx^2 + yy*dy^2) + xy*dx*dy
      - alpha = min(0.999, opacity * exp(-sigma))
      - Skip if sigma < 0 or alpha < 1/255
      - next_T <= 1e-4 terminates pixel traversal
    """
    G = len(g_indices)
    if G == 0 or px_flat is None:
        return {}, {}

    # Move all tensors to CPU and convert to Python lists for pure scalar arithmetic
    gi_list = [int(gi) for gi in g_indices.cpu()]

    # Extract parameters as flat Python arrays
    mx_list = means2d_0[g_indices, 0].cpu().tolist()
    my_list = means2d_0[g_indices, 1].cpu().tolist()
    xx_list = conics_0[g_indices, 0].cpu().tolist()
    xy_list = conics_0[g_indices, 1].cpu().tolist()
    yy_list = conics_0[g_indices, 2].cpu().tolist()
    opac_list = opacities[g_indices].cpu().tolist()

    px_list = px_flat.cpu().tolist()
    py_list = py_flat.cpu().tolist()

    wc = {gi: 0 for gi in gi_list}
    wu = {gi: 0 for gi in gi_list}

    for px, py in zip(px_list, py_list):
        T = 1.0
        for g_idx in range(G):
            gi = gi_list[g_idx]
            dx_p = px - mx_list[g_idx]
            dy_p = py - my_list[g_idx]
            s = 0.5 * (xx_list[g_idx] * dx_p * dx_p + yy_list[g_idx] * dy_p * dy_p) \
                + xy_list[g_idx] * dx_p * dy_p

            if s < 0.0:
                continue

            exp_s = math.exp(-s)
            alpha_r = opac_list[g_idx] * exp_s
            alpha_c = min(0.999, alpha_r)

            if alpha_c < 1.0 / 255.0:
                continue

            if T <= 1e-4:
                continue

            wc[gi] += 1
            if alpha_r <= 0.999:
                wu[gi] += 1

            T = T * (1.0 - alpha_c)

    return wc, wu


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="R3-0 W_it validation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--n-samples", type=int, default=128,
                        help="Number of random (tile, Gaussian) pairs to validate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    device = args.device
    tile_size = 16

    print("=" * 70)
    print("R3-0: W_it faithful replay validation")
    print("=" * 70)

    # ---- Load model ----
    config = ReferenceV1Config(scene="room", iterations=30000)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    N = model._xyz.shape[0]
    print(f"  Model loaded: N={N}")

    # ---- Load camera ----
    scene_dir = Path(REPO_ROOT) / "data" / "official" / "mipnerf360" / "room"
    cameras = load_cameras_from_json(str(scene_dir / "cameras.json"), device="cpu")
    cameras = resize_cameras(cameras, 1920, 1080)
    cam_sequence = np.load(os.path.join(REPO_ROOT, "data", "camera_sequence.npy"))
    cam_idx = int(cam_sequence[5000])
    cam = cameras[cam_idx]
    for attr in ["viewmatrix", "camera_center", "K"]:
        t = getattr(cam, attr, None)
        if isinstance(t, torch.Tensor):
            setattr(cam, attr, t.to(device))
    H, W = cam.image_height, cam.image_width
    print(f"  Camera: idx={cam_idx}, H={H}, W={W}")

    # ---- Project, SH, Intersect ----
    xyz = model.get_xyz
    quats = model.get_rotation
    scales = model.get_scaling
    opacities = model.get_opacity
    shs = model.get_features

    print("  Projecting...")
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        xyz, None, quats, scales,
        cam.viewmatrix.unsqueeze(0), cam.K.unsqueeze(0),
        W, H, eps2d=0.1
    )

    print("  SH...")
    dirs = xyz.unsqueeze(0) - cam.camera_center.unsqueeze(0)
    colors = spherical_harmonics(model.active_sh_degree, dirs, shs.unsqueeze(0))

    tile_w = (W + tile_size - 1) // tile_size
    tile_h = (H + tile_size - 1) // tile_size
    n_tiles = tile_h * tile_w

    print("  Intersect...")
    with torch.no_grad():
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_w, tile_h, sort=True
        )
        tile_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h)
    tile_offsets_flat = tile_offsets[0].reshape(-1)  # [n_tiles]

    print(f"  n_isects={flatten_ids.shape[0]} tiles={n_tiles}")
    print(f"  tile_h={tile_h} tile_w={tile_w}")

    # ---- Collect non-empty tiles ----
    non_empty = []
    for tile_idx in range(n_tiles):
        start = int(tile_offsets_flat[tile_idx])
        end = int(tile_offsets_flat[tile_idx + 1]) if tile_idx < n_tiles - 1 else len(flatten_ids)
        if end > start:
            non_empty.append(tile_idx)
    print(f"  Non-empty tiles: {len(non_empty)} / {n_tiles}")

    if len(non_empty) == 0:
        print("  ERROR: no non-empty tiles!")
        return 1

    # ---- Random sample of (tile, Gaussian) pairs ----
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    candidates = []
    for tile_idx in non_empty:
        ty = tile_idx // tile_w
        tx = tile_idx % tile_w
        start = int(tile_offsets_flat[tile_idx])
        end = int(tile_offsets_flat[tile_idx + 1]) if tile_idx < n_tiles - 1 else len(flatten_ids)
        g_indices = flatten_ids[start:end].long()
        g_unique = torch.unique(g_indices)
        for gi in g_unique:
            candidates.append((tile_idx, tx, ty, int(gi)))

    print(f"  Total (tile, Gaussian) candidates: {len(candidates)}")

    n_samples = min(args.n_samples, len(candidates))
    sampled = random.sample(candidates, n_samples)
    print(f"  Sampling {n_samples} pairs\n")

    # ---- Validate each sampled pair ----
    mismatches_color = 0
    mismatches_unclamped = 0
    max_wc = 0
    max_wu = 0
    detailed_results = []
    
    # Number of pixels to sample for scalar reference (tractable)
    N_SCALAR_PIXELS = 32

    for idx_in_sample, (tile_idx, tx, ty, gi) in enumerate(sampled):
        start = int(tile_offsets_flat[tile_idx])
        end = int(tile_offsets_flat[tile_idx + 1]) if tile_idx < n_tiles - 1 else len(flatten_ids)
        g_indices = flatten_ids[start:end].long()
        tile_x = tx * tile_size
        tile_y = ty * tile_size

        # Pre-compute pixel coordinates for this tile
        # For scalar comparison, use a random subsample (same pixels for both)
        scalar_px, scalar_py = _get_tile_pixels(
            tile_x, tile_y, tile_size, H, W, device,
            n_subsample=N_SCALAR_PIXELS
        )

        # GPU on full tile (for distribution stats)
        full_px, full_py = _get_tile_pixels(
            tile_x, tile_y, tile_size, H, W, device,
            n_subsample=None
        )
        wc_full, wu_full = compute_w_gpu(
            g_indices, means2d[0], conics[0], opacities,
            full_px, full_py
        )
        wc_g = wc_full.get(gi, 0)
        wu_g = wu_full.get(gi, 0)

        # GPU and scalar on SAME subsample for comparison
        wc_sub_gpu, wu_sub_gpu = compute_w_gpu(
            g_indices, means2d[0], conics[0], opacities,
            scalar_px, scalar_py
        )
        wc_sub_s, wu_sub_s = compute_w_scalar(
            g_indices, means2d[0], conics[0], opacities,
            scalar_px, scalar_py
        )
        wc_s_val = wc_sub_s.get(gi, 0) if wc_sub_s else 0
        wu_s_val = wu_sub_s.get(gi, 0) if wu_sub_s else 0
        wc_g_val = wc_sub_gpu.get(gi, 0) if wc_sub_gpu else 0
        wu_g_val = wu_sub_gpu.get(gi, 0) if wu_sub_gpu else 0

        mc = (wc_g_val != wc_s_val)
        mu = (wu_g_val != wu_s_val)
        if mc:
            mismatches_color += 1
        if mu:
            mismatches_unclamped += 1

        max_wc = max(max_wc, wc_g, wc_s_val)
        max_wu = max(max_wu, wu_g, wu_s_val)

        if mc or mu or idx_in_sample < 10:
            detailed_results.append({
                "sample": idx_in_sample,
                "tile_idx": tile_idx,
                "gaussian_id": gi,
                "w_color_gpu_full": wc_g,
                "w_color_gpu_sub": wc_g_val,
                "w_color_scalar": wc_s_val,
                "w_unclamped_gpu_full": wu_g,
                "w_unclamped_gpu_sub": wu_g_val,
                "w_unclamped_scalar": wu_s_val,
                "n_scalar_pixels": N_SCALAR_PIXELS,
                "match_color": not mc,
                "match_unclamped": not mu,
            })

    # Distribution stats from FULL GPU (all pixels)
    all_wc_vals = []
    all_wu_vals = []
    for tile_idx, tx, ty, gi in sampled:
        start = int(tile_offsets_flat[tile_idx])
        end = int(tile_offsets_flat[tile_idx + 1]) if tile_idx < n_tiles - 1 else len(flatten_ids)
        g_indices = flatten_ids[start:end].long()
        tile_x = tx * tile_size
        tile_y = ty * tile_size
        full_px, full_py = _get_tile_pixels(tile_x, tile_y, tile_size, H, W, device, n_subsample=None)
        wc_dict, wu_dict = compute_w_gpu(
            g_indices, means2d[0], conics[0], opacities,
            full_px, full_py
        )
        all_wc_vals.append(wc_dict.get(gi, 0) if wc_dict else 0)
        all_wu_vals.append(wu_dict.get(gi, 0) if wu_dict else 0)

    all_wc_arr = np.array(all_wc_vals, dtype=np.float64)
    all_wu_arr = np.array(all_wu_vals, dtype=np.float64)

    # ---- Report ----
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    def dist_stats(arr, label):
        if len(arr) == 0:
            print(f"  {label}: no data")
            return {}
        nz = np.count_nonzero(arr > 0)
        stats = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "fraction_zero": float(np.mean(arr == 0)),
            "fraction_one": float(np.mean(arr == 1)),
            "fraction_gt1": float(np.mean(arr > 1)),
        }
        print(f"  {label}:")
        print(f"    min={stats['min']:.1f}  median={stats['median']:.1f}  "
              f"p90={stats['p90']:.1f}  p99={stats['p99']:.1f}  max={stats['max']:.1f}")
        print(f"    zero={stats['fraction_zero']:.4f}  one={stats['fraction_one']:.4f}  "
              f">1={stats['fraction_gt1']:.4f}  nonzero={nz}/{len(arr)}")
        return stats

    wc_dist = dist_stats(all_wc_arr, "W_color")
    wu_dist = dist_stats(all_wu_arr, "W_unclamped")

    print(f"\n  MISMATCHES: W_color={mismatches_color}/{n_samples}  "
          f"W_unclamped={mismatches_unclamped}/{n_samples}")
    print(f"  Max W_color={max_wc}  Max W_unclamped={max_wu}")

    # Invariant checks
    all_ok = True

    if mismatches_color > 0:
        print(f"\n  *** FAIL: W_COLOR_MISMATCHES={mismatches_color} ***")
        all_ok = False
    else:
        print(f"\n  W_COLOR_MISMATCHES = 0  ✓")

    if mismatches_unclamped > 0:
        print(f"  *** FAIL: W_UNCLAMPED_MISMATCHES={mismatches_unclamped} ***")
        all_ok = False
    else:
        print(f"  W_UNCLAMPED_MISMATCHES = 0  ✓")

    # Check invariant: 0 <= W_unclamped <= W_color <= 256
    invariant_ok = np.all((all_wu_arr >= 0) & (all_wu_arr <= all_wc_arr) & (all_wc_arr <= 256))
    if not invariant_ok:
        print(f"  *** FAIL: invariant 0 <= W_unclamped <= W_color <= 256 violated ***")
        all_ok = False
    else:
        print(f"  Invariant 0 <= W_unclamped <= W_color <= 256 holds for all pairs  ✓")

    verdict = "PASS" if all_ok else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    # ---- Detailed pairs (first 10 + mismatches) ----
    print("\n  Detail (first 10 sampled + mismatches):")
    for d in detailed_results:
        marker = ""
        if not d["match_color"]:
            marker += " [WC-MISMATCH]"
        if not d["match_unclamped"]:
            marker += " [WU-MISMATCH]"
        print(f"    [{d['sample']}] tile={d['tile_idx']} gauss={d['gaussian_id']}: "
              f"wc_gpu={d['w_color_gpu_full']} wc_scalar={d['w_color_scalar']} "
              f"wu_gpu={d['w_unclamped_gpu_full']} wu_scalar={d['w_unclamped_scalar']}"
              f"  (sub: wc={d['w_color_gpu_sub']} wu={d['w_unclamped_gpu_sub']})"
              f"{marker}")

    # ---- Save JSON ----
    if args.output:
        result = {
            "R3_W_VALIDATION": verdict,
            "W_COLOR_MISMATCHES": mismatches_color,
            "W_UNCLAMPED_MISMATCHES": mismatches_unclamped,
            "n_sampled_pairs": n_samples,
            "W_COLOR_DISTRIBUTION": wc_dist,
            "W_UNCLAMPED_DISTRIBUTION": wu_dist,
            "invariant_holds": bool(invariant_ok),
            "detailed_pairs": detailed_results,
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved to {args.output}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
