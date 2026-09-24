#!/usr/bin/env python3
"""
C42 Phase 2-C: D-SSIM Pixel Contribution Profiling.

Measures the spatial distribution of D-SSIM contributions:
1. Full SSIM map from rendered vs GT
2. Per-pixel loss contribution (1 - SSIM)
3. Distribution analysis: what fraction of pixels contribute what fraction of loss
4. Spatial concentration: are high-contribution pixels clustered?
5. Predictability: can local variance/gradient identify high-contribution pixels?
6. Multi-camera analysis: does the pattern hold across different viewpoints?
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset

DEVICE = "cuda"
WINDOW = 11
SIGMA = 1.5
TILE_SIZE = 16
PACKED = True
SH_DEGREE = 3
EPS2D = 0.1
RADIUS_CLIP = 0.0


def compute_ssim_map(pred, target, window_size=WINDOW, sigma=SIGMA):
    """Return full SSIM map [1,3,H,W] and per-pixel loss contribution (1-SSIM)."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)

    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    def blur(x):
        return F.conv2d(x, kernel, padding=window_size // 2, groups=pred.shape[1])

    mu_pred = blur(pred)
    mu_target = blur(target)
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    sigma_pred_sq = blur(pred ** 2) - mu_pred_sq
    sigma_target_sq = blur(target ** 2) - mu_target_sq
    sigma_pred_target = blur(pred * target) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    loss_contrib = 1.0 - ssim_map  # [1,3,H,W]
    return ssim_map, loss_contrib


def analyze_camera(model, cam, gt_img, cam_name, n_patches=16):
    """Analyze D-SSIM contribution for one camera view."""
    img_w, img_h = cam.image_width, cam.image_height

    with torch.no_grad():
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        pred = rendered[0].clamp(0, 1)

    ssim_map, loss_contrib = compute_ssim_map(pred, gt_img)
    # loss_contrib: [1,3,H,W], average over channels → [H,W]
    contrib_avg = loss_contrib.mean(dim=1).squeeze(0)  # [H,W]
    ssim_avg = ssim_map.mean(dim=1).squeeze(0)  # [H,W]

    # Also compute L1 per-pixel for comparison
    l1_map = (pred - gt_img).abs().mean(dim=-1)  # [H,W]

    # ── Distribution analysis ──
    contrib_flat = contrib_avg.flatten()
    total_loss = contrib_flat.sum()
    n_pixels = contrib_flat.numel()

    # Sort by contribution (descending)
    contrib_sorted, indices = torch.sort(contrib_flat, descending=True)
    cumsum = torch.cumsum(contrib_sorted, dim=0)
    cumfrac = cumsum / total_loss

    # Find what fraction of pixels account for X% of loss
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    pixel_fracs = {}
    for thresh in thresholds:
        idx = torch.searchsorted(cumfrac, torch.tensor(thresh, device=DEVICE))
        frac = float((idx + 1) / n_pixels)
        pixel_fracs[f"{int(thresh*100)}pct_loss"] = frac

    # ── Statistics ──
    stats = {
        "camera": cam_name,
        "n_pixels": n_pixels,
        "mean_ssim": float(ssim_avg.mean()),
        "mean_contrib": float(contrib_avg.mean()),
        "std_contrib": float(contrib_avg.std()),
        "min_contrib": float(contrib_avg.min()),
        "max_contrib": float(contrib_avg.max()),
        "median_contrib": float(contrib_avg.median()),
        "p90_contrib": float(contrib_avg.flatten().quantile(0.90)),
        "p95_contrib": float(contrib_avg.flatten().quantile(0.95)),
        "p99_contrib": float(contrib_avg.flatten().quantile(0.99)),
        "frac_contrib_zero": float((contrib_avg < 1e-4).float().mean()),
        "frac_contrib_low": float((contrib_avg < 0.01).float().mean()),
        "frac_contrib_high": float((contrib_avg > 0.1).float().mean()),
        "pixel_fracs": pixel_fracs,
    }

    # ── Patch analysis (n_patches x n_patches grid) ──
    ph = img_h // n_patches
    pw = img_w // n_patches
    patch_contribs = np.zeros((n_patches, n_patches))
    patch_l1 = np.zeros((n_patches, n_patches))
    patch_variance = np.zeros((n_patches, n_patches))

    contrib_np = contrib_avg.cpu().numpy()
    l1_np = l1_map.cpu().numpy()
    pred_np = pred.cpu().numpy()

    for i in range(n_patches):
        for j in range(n_patches):
            patch = contrib_np[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            patch_contribs[i, j] = patch.mean()
            patch_l1[i, j] = l1_np[i*ph:(i+1)*ph, j*pw:(j+1)*pw].mean()
            # Local variance of rendered image (predictor for structural content)
            patch_img = pred_np[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
            patch_variance[i, j] = patch_img.var()

    # Correlation between patch variance and patch contribution
    var_flat = patch_variance.flatten()
    contrib_flat_patch = patch_contribs.flatten()
    l1_flat_patch = patch_l1.flatten()

    # Pearson correlation
    def pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    stats["patch_corr_variance_contrib"] = pearson(var_flat, contrib_flat_patch)
    stats["patch_corr_l1_contrib"] = pearson(l1_flat_patch, contrib_flat_patch)
    stats["patch_corr_variance_l1"] = pearson(var_flat, l1_flat_patch)

    # ── Spatial concentration (Moran's I simplified) ──
    # Check if high-contribution pixels are spatially clustered
    high_mask = (contrib_avg > contrib_avg.median()).float().cpu().numpy()
    # Simple measure: fraction of 4-neighbors that are also high
    neighbor_same = 0.0
    neighbor_total = 0.0
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        shifted = np.roll(high_mask, (dy, dx), axis=(0, 1))
        neighbor_same += (high_mask * shifted).sum()
        neighbor_total += high_mask.sum()
    spatial_clustering = float(neighbor_same / (neighbor_total * 4)) if neighbor_total > 0 else 0
    stats["spatial_clustering"] = spatial_clustering  # 0.5=random, 1.0=clustered

    return stats, contrib_avg, ssim_avg, l1_map, pred


def main():
    print("=" * 72)
    print("C42 Phase 2-C: D-SSIM Pixel Contribution Profiling")
    print("=" * 72)

    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parent.parent.parent

    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)

    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=DEVICE)
    N = model.xyz.shape[0]
    print(f"  Model: {N:,} Gaussians")

    # Analyze multiple cameras
    cam_indices = [0, 50, 100, 150, 200, 250, 300]
    all_stats = []

    for cam_idx in cam_indices:
        cam = dataset.get_camera(cam_idx)
        gt_img = dataset.get_gt_image(cam_idx)
        cam_name = f"cam_{cam_idx:03d}"
        print(f"\n  Analyzing {cam_name}...")

        stats, contrib, ssim, l1, pred = analyze_camera(model, cam, gt_img, cam_name)
        all_stats.append(stats)

        print(f"    Mean SSIM: {stats['mean_ssim']:.4f}  Mean contrib: {stats['mean_contrib']:.6f}")
        print(f"    Contrib distribution: min={stats['min_contrib']:.6f}  median={stats['median_contrib']:.6f}  "
              f"p90={stats['p90_contrib']:.6f}  p99={stats['p99_contrib']:.6f}  max={stats['max_contrib']:.6f}")
        print(f"    Frac contrib < 1e-4: {stats['frac_contrib_zero']:.1%}")
        print(f"    Frac contrib < 0.01: {stats['frac_contrib_low']:.1%}")
        print(f"    Frac contrib > 0.1:  {stats['frac_contrib_high']:.1%}")
        print(f"    Pixel fracs for loss thresholds:")
        for k, v in stats["pixel_fracs"].items():
            print(f"      {k}: {v:.1%} of pixels")
        print(f"    Patch corr (variance vs contrib): {stats['patch_corr_variance_contrib']:.3f}")
        print(f"    Patch corr (L1 vs contrib):       {stats['patch_corr_l1_contrib']:.3f}")
        print(f"    Spatial clustering: {stats['spatial_clustering']:.3f} (0.5=random, 1.0=clustered)")

    # ── Cross-camera summary ──
    print(f"\n{'='*72}")
    print("CROSS-CAMERA SUMMARY")
    print(f"{'='*72}")

    avg_fracs_low = np.mean([s["frac_contrib_low"] for s in all_stats])
    avg_fracs_zero = np.mean([s["frac_contrib_zero"] for s in all_stats])
    avg_fracs_high = np.mean([s["frac_contrib_high"] for s in all_stats])
    avg_corr_var = np.mean([s["patch_corr_variance_contrib"] for s in all_stats])
    avg_corr_l1 = np.mean([s["patch_corr_l1_contrib"] for s in all_stats])
    avg_clustering = np.mean([s["spatial_clustering"] for s in all_stats])

    avg_pixel_fracs = {}
    for k in all_stats[0]["pixel_fracs"]:
        avg_pixel_fracs[k] = np.mean([s["pixel_fracs"][k] for s in all_stats])

    print(f"\n  Avg frac contrib < 1e-4 (near-zero): {avg_fracs_zero:.1%}")
    print(f"  Avg frac contrib < 0.01 (low):       {avg_fracs_low:.1%}")
    print(f"  Avg frac contrib > 0.1 (high):       {avg_fracs_high:.1%}")
    print(f"\n  Avg pixels needed for X% of loss:")
    for k, v in avg_pixel_fracs.items():
        print(f"    {k}: {v:.1%} of pixels")
    print(f"\n  Avg patch corr (local variance → contrib): {avg_corr_var:.3f}")
    print(f"  Avg patch corr (L1 error → contrib):       {avg_corr_l1:.3f}")
    print(f"  Avg spatial clustering: {avg_clustering:.3f}")

    # ── Hypothesis evaluation ──
    print(f"\n{'='*72}")
    print("HYPOTHESIS EVALUATION")
    print(f"{'='*72}")

    print(f"\n  H1: Most pixels have near-zero D-SSIM contribution")
    print(f"    Evidence: {avg_fracs_low:.1%} of pixels have contrib < 0.01")
    print(f"    Evidence: {avg_fracs_zero:.1%} of pixels have contrib < 1e-4")
    if avg_fracs_low > 0.5:
        print(f"    → SUPPORTED: majority of pixels contribute negligibly")
    else:
        print(f"    → PARTIALLY SUPPORTED: significant fraction but not majority")

    print(f"\n  H2: A small fraction of pixels accounts for most of the loss")
    print(f"    Evidence: {avg_pixel_fracs.get('90pct_loss', 0):.1%} of pixels → 90% of loss")
    print(f"    Evidence: {avg_pixel_fracs.get('50pct_loss', 0):.1%} of pixels → 50% of loss")
    if avg_pixel_fracs.get('90pct_loss', 1) < 0.5:
        print(f"    → SUPPORTED: <50% of pixels produce 90% of loss")
    else:
        print(f"    → WEAK: loss is relatively uniformly distributed")

    print(f"\n  H3: High-contribution pixels are spatially clustered")
    print(f"    Evidence: spatial clustering = {avg_clustering:.3f} (0.5=random, 1.0=perfect)")
    if avg_clustering > 0.7:
        print(f"    → SUPPORTED: high-contribution pixels are spatially concentrated")
    else:
        print(f"    → NOT SUPPORTED: contributions are spatially dispersed")

    print(f"\n  H4: Local image variance predicts D-SSIM contribution")
    print(f"    Evidence: patch corr = {avg_corr_var:.3f}")
    if abs(avg_corr_var) > 0.5:
        print(f"    → SUPPORTED: variance is a good predictor")
    elif abs(avg_corr_var) > 0.3:
        print(f"    → MODERATE: variance is a partial predictor")
    else:
        print(f"    → NOT SUPPORTED: variance does not predict contribution")

    print(f"\n  H5: L1 error predicts D-SSIM contribution")
    print(f"    Evidence: patch corr = {avg_corr_l1:.3f}")
    if abs(avg_corr_l1) > 0.5:
        print(f"    → SUPPORTED: L1 is a good predictor")
    elif abs(avg_corr_l1) > 0.3:
        print(f"    → MODERATE: L1 is a partial predictor")
    else:
        print(f"    → NOT SUPPORTED: L1 does not predict contribution")

    # ── Save ──
    output = {
        "experiment": "C42 Phase 2-C: D-SSIM Pixel Contribution Profiling",
        "config": {"scene": "room", "n_gaussians": N, "window": WINDOW, "sigma": SIGMA,
                   "cameras_analyzed": cam_indices},
        "per_camera_stats": all_stats,
        "cross_camera_summary": {
            "avg_frac_contrib_zero": float(avg_fracs_zero),
            "avg_frac_contrib_low": float(avg_fracs_low),
            "avg_frac_contrib_high": float(avg_fracs_high),
            "avg_pixel_fracs": {k: float(v) for k, v in avg_pixel_fracs.items()},
            "avg_corr_variance_contrib": float(avg_corr_var),
            "avg_corr_l1_contrib": float(avg_corr_l1),
            "avg_spatial_clustering": float(avg_clustering),
        },
    }

    save_path = Path("results/phase-c42/c42_pixel_contrib_data.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
