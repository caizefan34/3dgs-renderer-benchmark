#!/usr/bin/env python3
"""
C42 Phase 2-C: Temporal evolution of D-SSIM pixel contribution.

Tests whether D-SSIM contributions become spatially concentrated
as the model converges from iter 5000 to 30000.

Checkpoints: 5000, 10000, 15000, 20000, 25000, 30000
Key metrics per checkpoint:
  - Mean SSIM (quality level)
  - Pixel fraction for 90% of loss (concentration)
  - Spatial clustering (contiguity)
  - L1 correlation (predictability)
  - Fraction of near-zero contribution pixels
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
EPS2D = 0.1
RADIUS_CLIP = 0.0

CKPT_BASE = "results/epic05/phase7/phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter{}.pt"
ITERATIONS = [5000, 10000, 15000, 20000, 25000, 30000]


def compute_ssim_map(pred, target, window_size=WINDOW, sigma=SIGMA):
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
    return ssim_map, 1.0 - ssim_map


def analyze_checkpoint(ckpt_path, dataset, cam_indices):
    """Analyze D-SSIM contribution for multiple cameras at one checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=DEVICE)
    N = model.xyz.shape[0]

    results = []
    for cam_idx in cam_indices:
        cam = dataset.get_camera(cam_idx)
        gt_img = dataset.get_gt_image(cam_idx)
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
        contrib_avg = loss_contrib.mean(dim=1).squeeze(0)  # [H,W]
        ssim_avg = ssim_map.mean(dim=1).squeeze(0)
        l1_map = (pred - gt_img).abs().mean(dim=-1)  # [H,W]

        n_pixels = contrib_avg.numel()
        contrib_flat = contrib_avg.flatten()
        total_loss = contrib_flat.sum()

        # Concentration: pixel fraction for X% of loss
        contrib_sorted, _ = torch.sort(contrib_flat, descending=True)
        cumsum = torch.cumsum(contrib_sorted, dim=0)
        cumfrac = cumsum / total_loss

        frac_50 = float(torch.searchsorted(cumfrac, torch.tensor(0.50, device=DEVICE)) / n_pixels)
        frac_90 = float(torch.searchsorted(cumfrac, torch.tensor(0.90, device=DEVICE)) / n_pixels)
        frac_95 = float(torch.searchsorted(cumfrac, torch.tensor(0.95, device=DEVICE)) / n_pixels)

        # Fraction of low/near-zero contribution pixels
        frac_low = float((contrib_avg < 0.01).float().mean())
        frac_zero = float((contrib_avg < 1e-4).float().mean())

        # Spatial clustering
        high_mask = (contrib_avg > contrib_avg.median()).float().cpu().numpy()
        neighbor_same = 0.0
        neighbor_total = 0.0
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = np.roll(high_mask, (dy, dx), axis=(0, 1))
            neighbor_same += (high_mask * shifted).sum()
            neighbor_total += high_mask.sum()
        clustering = float(neighbor_same / (neighbor_total * 4)) if neighbor_total > 0 else 0

        # Patch-level L1 correlation
        n_patches = 16
        ph, pw = img_h // n_patches, img_w // n_patches
        patch_contrib = np.zeros((n_patches, n_patches))
        patch_l1 = np.zeros((n_patches, n_patches))
        contrib_np = contrib_avg.cpu().numpy()
        l1_np = l1_map.cpu().numpy()
        for i in range(n_patches):
            for j in range(n_patches):
                patch_contrib[i, j] = contrib_np[i*ph:(i+1)*ph, j*pw:(j+1)*pw].mean()
                patch_l1[i, j] = l1_np[i*ph:(i+1)*ph, j*pw:(j+1)*pw].mean()

        def pearson(a, b):
            a, b = a - a.mean(), b - b.mean()
            d = np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())
            return float(np.dot(a, b) / d) if d > 0 else 0.0

        corr_l1 = pearson(patch_l1.flatten(), patch_contrib.flatten())

        # PSNR
        mse = float(((pred - gt_img) ** 2).mean())
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        results.append({
            "cam_idx": cam_idx,
            "n_gaussians": N,
            "psnr": psnr,
            "mean_ssim": float(ssim_avg.mean()),
            "mean_contrib": float(contrib_avg.mean()),
            "frac_50pct_loss": frac_50,
            "frac_90pct_loss": frac_90,
            "frac_95pct_loss": frac_95,
            "frac_low_contrib": frac_low,
            "frac_zero_contrib": frac_zero,
            "spatial_clustering": clustering,
            "corr_l1_contrib": corr_l1,
        })

    # Average across cameras
    avg = {
        "n_gaussians": N,
        "avg_psnr": float(np.mean([r["psnr"] for r in results])),
        "avg_ssim": float(np.mean([r["mean_ssim"] for r in results])),
        "avg_frac_50pct": float(np.mean([r["frac_50pct_loss"] for r in results])),
        "avg_frac_90pct": float(np.mean([r["frac_90pct_loss"] for r in results])),
        "avg_frac_95pct": float(np.mean([r["frac_95pct_loss"] for r in results])),
        "avg_frac_low": float(np.mean([r["frac_low_contrib"] for r in results])),
        "avg_frac_zero": float(np.mean([r["frac_zero_contrib"] for r in results])),
        "avg_clustering": float(np.mean([r["spatial_clustering"] for r in results])),
        "avg_corr_l1": float(np.mean([r["corr_l1_contrib"] for r in results])),
    }
    return results, avg


def main():
    print("=" * 72)
    print("C42 Phase 2-C: Temporal Evolution of D-SSIM Contribution")
    print("=" * 72)

    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parent.parent.parent

    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    cam_indices = [0, 50, 100, 150, 200, 250, 300]

    all_checkpoints = []

    print(f"\n{'Iter':>6s}  {'GS':>8s}  {'PSNR':>6s}  {'SSIM':>6s}  "
          f"{'F90%':>6s}  {'F50%':>6s}  {'Flow':>6s}  {'Fzero':>6s}  "
          f"{'Clust':>6s}  {'CorrL1':>7s}")
    print("-" * 85)

    for it in ITERATIONS:
        ckpt_path = repo_root / CKPT_BASE.format(it)
        if not ckpt_path.exists():
            print(f"  {it:>6d}  -- checkpoint not found --")
            continue

        print(f"  Loading iter {it}...", end=" ", flush=True)
        per_cam, avg = analyze_checkpoint(ckpt_path, dataset, cam_indices)
        all_checkpoints.append({"iteration": it, **avg, "per_camera": per_cam})
        print("done")

        print(f"  {it:>6d}  {avg['n_gaussians']:>8,}  {avg['avg_psnr']:>6.2f}  {avg['avg_ssim']:>6.4f}  "
              f"{avg['avg_frac_90pct']:>6.1%}  {avg['avg_frac_50pct']:>6.1%}  "
              f"{avg['avg_frac_low']:>6.1%}  {avg['avg_frac_zero']:>6.1%}  "
              f"{avg['avg_clustering']:>6.3f}  {avg['avg_corr_l1']:>7.3f}")

    # ── Temporal analysis ──
    print(f"\n{'='*72}")
    print("TEMPORAL ANALYSIS")
    print(f"{'='*72}")

    if len(all_checkpoints) >= 2:
        first = all_checkpoints[0]
        last = all_checkpoints[-1]
        print(f"\n  Evolution from iter {first['iteration']} to iter {last['iteration']}:")
        print(f"    PSNR:            {first['avg_psnr']:.2f} → {last['avg_psnr']:.2f}  ({last['avg_psnr']-first['avg_psnr']:+.2f} dB)")
        print(f"    SSIM:            {first['avg_ssim']:.4f} → {last['avg_ssim']:.4f}  ({last['avg_ssim']-first['avg_ssim']:+.4f})")
        print(f"    Frac for 90%:    {first['avg_frac_90pct']:.1%} → {last['avg_frac_90pct']:.1%}")
        print(f"    Frac for 50%:    {first['avg_frac_50pct']:.1%} → {last['avg_frac_50pct']:.1%}")
        print(f"    Frac low contrib:{first['avg_frac_low']:.1%} → {last['avg_frac_low']:.1%}")
        print(f"    Frac zero contrib:{first['avg_frac_zero']:.1%} → {last['avg_frac_zero']:.1%}")
        print(f"    Clustering:      {first['avg_clustering']:.3f} → {last['avg_clustering']:.3f}")
        print(f"    L1 correlation:  {first['avg_corr_l1']:.3f} → {last['avg_corr_l1']:.3f}")

    # ── Hypothesis evaluation ──
    print(f"\n{'='*72}")
    print("HYPOTHESIS EVALUATION (with temporal dimension)")
    print(f"{'='*72}")

    if last:
        print(f"\n  H1: D-SSIM contributions become concentrated as model converges")
        print(f"    iter {first['iteration']}: {first['avg_frac_90pct']:.1%} pixels → 90% loss")
        print(f"    iter {last['iteration']}: {last['avg_frac_90pct']:.1%} pixels → 90% loss")
        if last['avg_frac_90pct'] < first['avg_frac_90pct'] * 0.7:
            print(f"    → SUPPORTED: concentration increased by {(1-last['avg_frac_90pct']/first['avg_frac_90pct'])*100:.0f}%")
        else:
            print(f"    → WEAK: concentration did not significantly increase")

        print(f"\n  H2: Low-contribution pixels emerge as model converges")
        print(f"    iter {first['iteration']}: {first['avg_frac_low']:.1%} pixels have contrib < 0.01")
        print(f"    iter {last['iteration']}: {last['avg_frac_low']:.1%} pixels have contrib < 0.01")
        if last['avg_frac_low'] > 0.1:
            print(f"    → SUPPORTED: {last['avg_frac_low']:.1%} of pixels are low-contribution at convergence")
        else:
            print(f"    → NOT SUPPORTED: even at convergence, low-contribution fraction is {last['avg_frac_low']:.1%}")

        print(f"\n  H3: Spatial clustering increases with convergence")
        print(f"    iter {first['iteration']}: clustering = {first['avg_clustering']:.3f}")
        print(f"    iter {last['iteration']}: clustering = {last['avg_clustering']:.3f}")
        if last['avg_clustering'] > first['avg_clustering'] * 1.2:
            print(f"    → SUPPORTED: clustering increased")
        else:
            print(f"    → NOT SUPPORTED: clustering unchanged or decreased")

        print(f"\n  H4: L1 error becomes better predictor at convergence")
        print(f"    iter {first['iteration']}: corr = {first['avg_corr_l1']:.3f}")
        print(f"    iter {last['iteration']}: corr = {last['avg_corr_l1']:.3f}")
        if last['avg_corr_l1'] > 0.7:
            print(f"    → SUPPORTED: L1 is a strong predictor at convergence")
        elif last['avg_corr_l1'] > 0.5:
            print(f"    → MODERATE: L1 is a useful predictor at convergence")
        else:
            print(f"    → NOT SUPPORTED: L1 prediction does not improve")

    # ── Save ──
    output = {
        "experiment": "C42 Phase 2-C: Temporal Evolution of D-SSIM Contribution",
        "config": {"scene": "room", "iterations": ITERATIONS, "cameras": cam_indices},
        "checkpoints": all_checkpoints,
    }
    save_path = Path("results/phase-c42/c42_temporal_contrib_data.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
