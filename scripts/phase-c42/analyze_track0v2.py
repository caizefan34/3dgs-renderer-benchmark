#!/usr/bin/env python3
"""Analyze Track 0v2 results across all scales."""
import json, numpy as np
from pathlib import Path

results_dir = Path(__file__).parent.parent.parent / "results" / "a100" / "phase-c42"
scales = {"1.0": "10", "0.875": "0875", "0.75": "075", "0.625": "0625", "0.5": "05"}

print("=" * 80)
print("Track 0v2: C42 Re-validation with aggressive pruning")
print("=" * 80)
print(f"{'Scale':>8} {'GS_final':>10} {'PSNR':>8} {'SSIM':>8} {'MeanIter':>10} {'MeanSSIM_t':>12} {'MeanL1_t':>10}")
print("-" * 80)

all_data = {}
for scale_label, scale_str in scales.items():
    fpath = results_dir / f"track0v2_pruned_{scale_str}.json"
    if not fpath.exists():
        print(f"{scale_label:>8} MISSING")
        continue
    with open(fpath) as f:
        data = json.load(f)
    all_data[scale_label] = data

    iter_times = data["per_iter_times"]
    ssim_times = data["ssim_times"]
    l1_times = data["l1_times"]
    final = data["eval_points"][-1]

    # Skip first iter (warmup)
    mean_iter = np.mean(iter_times[1:]) if len(iter_times) > 1 else np.mean(iter_times)
    mean_ssim = np.mean(ssim_times)
    mean_l1 = np.mean(l1_times)

    print(f"{scale_label:>8} {final['gaussians']:>10,} {final['psnr']:>8.2f} {final['ssim']:>8.4f} {mean_iter:>10.1f} {mean_ssim:>12.2f} {mean_l1:>10.2f}")

# Compute C42 speedup
print(f"\n{'='*80}")
print("C42 Speedup Analysis (SSIM downscale savings)")
print(f"{'='*80}")
print(f"{'Scale':>8} {'GS_final':>10} {'MeanIter':>10} {'SSIM_time':>12} {'SSIM_savings':>14} {'E2E_speedup':>12}")
print("-" * 80)

baseline_iter = None
baseline_ssim = None
for scale_label in ["1.0", "0.875", "0.75", "0.625", "0.5"]:
    if scale_label not in all_data:
        continue
    data = all_data[scale_label]
    iter_times = data["per_iter_times"]
    ssim_times = data["ssim_times"]
    final = data["eval_points"][-1]
    mean_iter = np.mean(iter_times[1:]) if len(iter_times) > 1 else np.mean(iter_times)
    mean_ssim = np.mean(ssim_times)

    if scale_label == "1.0":
        baseline_iter = mean_iter
        baseline_ssim = mean_ssim
        ssim_savings = 0.0
        e2e_speedup = 0.0
    else:
        ssim_savings = baseline_ssim - mean_ssim
        e2e_speedup = (baseline_iter - mean_iter) / baseline_iter * 100

    print(f"{scale_label:>8} {final['gaussians']:>10,} {mean_iter:>10.1f} {mean_ssim:>12.2f} {ssim_savings:>14.2f} {e2e_speedup:>+11.1f}%")

# GS trajectory
print(f"\n{'='*80}")
print("GS Trajectory (every 500 iters)")
print(f"{'='*80}")
print(f"{'Iter':>6}", end="")
for scale_label in ["1.0", "0.875", "0.75", "0.625", "0.5"]:
    print(f" {scale_label:>12}", end="")
print()
for ep in range(0, len(all_data.get("1.0", {}).get("eval_points", []))):
    print(f"{ep*500:>6}", end="")
    for scale_label in ["1.0", "0.875", "0.75", "0.625", "0.5"]:
        if scale_label in all_data and ep < len(all_data[scale_label]["eval_points"]):
            gs = all_data[scale_label]["eval_points"][ep]["gaussians"]
            print(f" {gs:>12,}", end="")
        else:
            print(f" {'N/A':>12}", end="")
    print()

# Per-iter time trajectory (stable region: iter 2000+)
print(f"\n{'='*80}")
print("Per-iter time (stable region, iter 2000+)")
print(f"{'='*80}")
for scale_label in ["1.0", "0.875", "0.75", "0.625", "0.5"]:
    if scale_label not in all_data:
        continue
    iter_times = all_data[scale_label]["per_iter_times"]
    # iter 2000+ = index 20+
    stable = iter_times[20:] if len(iter_times) > 20 else iter_times
    print(f"  scale={scale_label}: mean={np.mean(stable):.1f} ms, std={np.std(stable):.1f} ms, n={len(stable)}")
