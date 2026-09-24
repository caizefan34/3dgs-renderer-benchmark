#!/usr/bin/env python3
"""Final 30K comparison analysis."""
import json
from pathlib import Path

d = Path("results/a100/phase-c45")

print("=" * 90)
print("30K Aggressive Pruning: Baseline vs Sep_freq8 Comparison")
print("=" * 90)

# The 30K results were saved with the same names as 5K, overwriting them
# We need to read from the 30k-specific files or use known values

# Aggressive 30K room baseline (just fetched as D_D_baseline_room_30k.json)
f_base = d / "D_D_baseline_room_30k.json"
if f_base.exists():
    data_base = json.load(open(f_base))
    if data_base.get("iters") == 30000:
        tt_base = data_base["total_train_time_s"]
        final_base = data_base["eval_points"][-1]
        psnrs_base = [ep["psnr"] for ep in data_base["eval_points"]]
        print(f"\nBaseline (orig SSIM, every iter, 30K aggressive):")
        print(f"  Time: {tt_base:.1f}s  Final PSNR: {final_base['psnr']:.2f}  SSIM: {final_base['ssim']:.4f}  GS: {final_base['gaussians']:,}")
        print(f"  Peak PSNR: {max(psnrs_base):.2f} at iter {data_base['eval_points'][psnrs_base.index(max(psnrs_base))]['iter']}")
        print(f"  Degradation: {max(psnrs_base) - psnrs_base[-1]:.2f} dB")

# Aggressive 30K room sep_freq8 (fetched earlier, overwritten by 30K)
# Known values from earlier analysis:
tt_sep = 909.4
final_psnr_sep = 24.23
final_ssim_sep = 0.8081
final_gs_sep = 1746432
peak_psnr_sep = 30.57

print(f"\nSep_freq8 (sep SSIM, every 8, 30K aggressive):")
print(f"  Time: {tt_sep:.1f}s  Final PSNR: {final_psnr_sep:.2f}  SSIM: {final_ssim_sep:.4f}  GS: {final_gs_sep:,}")
print(f"  Peak PSNR: {peak_psnr_sep:.2f} at iter 1000")
print(f"  Degradation: {peak_psnr_sep - final_psnr_sep:.2f} dB")

if f_base.exists() and data_base.get("iters") == 30000:
    speedup = (1 - tt_sep / tt_base) * 100
    psnr_diff = final_psnr_sep - final_base["psnr"]
    print(f"\n30K Comparison (aggressive pruning):")
    print(f"  Speedup: {speedup:.1f}%  ({tt_base/tt_sep:.2f}x faster)")
    print(f"  PSNR difference: {psnr_diff:+.2f} dB (sep_freq8 is {'BETTER' if psnr_diff > 0 else 'WORSE'})")
    print(f"  SSIM difference: {final_ssim_sep - final_base['ssim']:+.4f}")
    print(f"  GS difference: {final_gs_sep - final_base['gaussians']:+,}")

# 5K comparison (for reference)
print(f"\n{'=' * 90}")
print("5K Comparison (aggressive pruning, for reference)")
print(f"{'=' * 90}")
print(f"  Baseline:  506.9s  PSNR=24.58  SSIM=0.7992  GS=1,873,067")
print(f"  Sep_freq8: 115.3s  PSNR=25.11  SSIM=0.8291  GS=~1,500,000")
print(f"  Speedup:   +77.5%  (4.44x faster)")
print(f"  PSNR diff: +0.53 dB")

# Speedup scaling: 5K vs 30K
print(f"\n{'=' * 90}")
print("Speedup Scaling: 5K vs 30K")
print(f"{'=' * 90}")
if f_base.exists() and data_base.get("iters") == 30000:
    speedup_5k = (1 - 115.3 / 506.9) * 100
    speedup_30k = (1 - tt_sep / tt_base) * 100
    print(f"  5K speedup:  {speedup_5k:.1f}%  ({506.9/115.3:.2f}x)")
    print(f"  30K speedup: {speedup_30k:.1f}%  ({tt_base/tt_sep:.2f}x)")
    print(f"  Speedup is {'maintained' if abs(speedup_5k - speedup_30k) < 10 else 'changed'} at 30K")

# Moderate pruning progress
print(f"\n{'=' * 90}")
print("Moderate Pruning 30K Progress (from logs)")
print(f"{'=' * 90}")
print(f"  Baseline:  iter 8000  PSNR=24.27  GS=4,098,930  (still improving)")
print(f"  Sep_freq8: iter 26000 PSNR=25.13  GS=2,499,985  (almost done, stable)")
print(f"  Sep_ssim:  iter 13000 PSNR=23.85  GS=4,543,254  (stable)")
print(f"\n  Key: Moderate sep_freq8 at iter 26000 (PSNR=25.13) already exceeds:")
print(f"    - Aggressive sep_freq8 at 30K: 24.23")
print(f"    - Aggressive baseline at 30K: 23.59")
print(f"    - Moderate baseline at iter 8000: 24.27")
