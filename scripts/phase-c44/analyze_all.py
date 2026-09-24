#!/usr/bin/env python3
"""Analyze all Phase C44 training experiment results."""
import json, numpy as np
from pathlib import Path

results_dir = Path(__file__).parent.parent.parent / "results" / "a100" / "phase-c44"

experiments = {
    "baseline": "baseline_baseline.json",
    "A1": "A_A1.json",
    "A2": "A_A2.json",
    "A3": "A_A3.json",
    "C-freq2": "C_freq2.json",
    "C-freq4": "C_freq4.json",
    "C-freq8": "C_freq8.json",
}

all_data = {}
for label, fname in experiments.items():
    fpath = results_dir / fname
    if not fpath.exists():
        print(f"  {label}: MISSING")
        continue
    with open(fpath) as f:
        all_data[label] = json.load(f)

print("=" * 90)
print("Phase C44: Training Experiment Results")
print("=" * 90)
print(f"{'Experiment':<12} {'Total(s)':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'GS_final':>10} {'MeanIter':>10} {'MeanSSIM':>10}")
print("-" * 90)

for label in experiments:
    if label not in all_data:
        continue
    data = all_data[label]
    final = data["eval_points"][-1]
    iter_times = data["per_iter_times"][1:]
    ssim_times = data["ssim_times"]
    total = data["total_train_time_s"]
    print(f"{label:<12} {total:>8.1f} {final['psnr']:>8.2f} {final['ssim']:>8.4f} {final['lpips']:>8.4f} {final['gaussians']:>10,} {np.mean(iter_times):>10.1f} {np.mean(ssim_times):>10.2f}")

# Speedup vs baseline
baseline_data = all_data.get("baseline")
if baseline_data:
    baseline_time = baseline_data["total_train_time_s"]
    baseline_psnr = baseline_data["eval_points"][-1]["psnr"]
    baseline_ssim = baseline_data["eval_points"][-1]["ssim"]
    baseline_iter = np.mean(baseline_data["per_iter_times"][1:])
    
    print(f"\n{'='*90}")
    print("Speedup vs baseline")
    print(f"{'='*90}")
    print(f"{'Experiment':<12} {'Time(s)':>8} {'Speedup':>10} {'PSNR_diff':>10} {'SSIM_diff':>10} {'Decision':>10}")
    print("-" * 65)
    
    for label in experiments:
        if label not in all_data or label == "baseline":
            continue
        data = all_data[label]
        total = data["total_train_time_s"]
        final = data["eval_points"][-1]
        speedup = (baseline_time - total) / baseline_time * 100
        psnr_diff = final["psnr"] - baseline_psnr
        ssim_diff = final["ssim"] - baseline_ssim
        if speedup > 5 and psnr_diff > -0.5:
            decision = "KEEP"
        elif speedup < 3:
            decision = "DROP"
        elif psnr_diff < -1.0:
            decision = "DROP"
        else:
            decision = "REVIEW"
        print(f"{label:<12} {total:>8.1f} {speedup:>+9.1f}% {psnr_diff:>+10.2f} {ssim_diff:>+10.4f} {decision:>10}")

# Convergence curves
print(f"\n{'='*90}")
print("Convergence curves (PSNR at each eval point)")
print(f"{'='*90}")
print(f"{'Iter':<6}", end="")
for label in experiments:
    if label in all_data:
        print(f" {label:>10}", end="")
print()
for i in range(len(all_data.get("baseline", {}).get("eval_points", []))):
    print(f"{i*500:<6}", end="")
    for label in experiments:
        if label not in all_data:
            continue
        if i < len(all_data[label]["eval_points"]):
            psnr = all_data[label]["eval_points"][i]["psnr"]
            print(f" {psnr:>10.2f}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

# SSIM time comparison
print(f"\n{'='*90}")
print("SSIM time per iteration (mean)")
print(f"{'='*90}")
for label in experiments:
    if label not in all_data:
        continue
    ssim_times = all_data[label]["ssim_times"]
    # Only count non-zero SSIM times (Track C has zeros on non-SSIM iters)
    nonzero = [t for t in ssim_times if t > 0.01]
    zero_count = len(ssim_times) - len(nonzero)
    if nonzero:
        print(f"  {label:<12}: mean_ssim={np.mean(nonzero):.2f}ms (when used), zero_iters={zero_count}/{len(ssim_times)}")
    else:
        print(f"  {label:<12}: no SSIM timing data")
