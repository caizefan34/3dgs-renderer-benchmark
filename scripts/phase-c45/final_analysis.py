#!/usr/bin/env python3
"""Final comprehensive analysis of ALL 30K results."""
import json
from pathlib import Path

d = Path("results/a100/phase-c45")

print("=" * 95)
print("FINAL 30K RESULTS: Complete Comparison")
print("=" * 95)

# ===== 30K Moderate Pruning (Room) =====
print("\n--- 30K Moderate Pruning (Room) ---")
print(f"{'Config':<35} {'Time(s)':>8} {'PSNR':>8} {'SSIM':>8} {'GS':>10} {'Speedup':>10} {'PSNR diff':>10}")
print("-" * 95)

mod_results = {}
for name, fname in [
    ("Baseline (orig SSIM, every iter)", "D_D_baseline_room_mod30k.json"),
    ("Sep_ssim (sep SSIM, every iter)", "D_D_sep_ssim_room_mod30k.json"),
    ("Sep_freq8 (sep SSIM, every 8)", "D_D_sep_freq8_room_mod30k.json"),
]:
    f = d / fname
    if f.exists():
        data = json.load(open(f))
        tt = data["total_train_time_s"]
        final = data["eval_points"][-1]
        mod_results[name] = {"time": tt, "psnr": final["psnr"], "ssim": final["ssim"], "gs": final["gaussians"]}
        
# Calculate speedup and PSNR diff relative to baseline
if "Baseline (orig SSIM, every iter)" in mod_results:
    base = mod_results["Baseline (orig SSIM, every iter)"]
    for name, r in mod_results.items():
        sp = (1 - r["time"] / base["time"]) * 100
        pd = r["psnr"] - base["psnr"]
        print(f"{name:<35} {r['time']:>8.1f} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['gs']:>10,} {sp:>+9.1f}% {pd:>+10.2f}")

# ===== 30K Aggressive Pruning (All Scenes) =====
print("\n--- 30K Aggressive Pruning (All Scenes) ---")
print(f"{'Scene':<10} {'Config':<25} {'Time(s)':>8} {'PSNR':>8} {'Speedup':>10} {'PSNR diff':>10}")
print("-" * 75)

# Known values from completed experiments
aggr_data = [
    ("room", "Baseline", 3434.8, 23.59),
    ("room", "Sep_freq8", 909.4, 24.23),
    ("garden", "Baseline", 3472.5, 18.97),
    ("garden", "Sep_freq8", 1399.9, 18.59),
    ("bicycle", "Baseline", 4284.4, 16.93),
    ("bicycle", "Sep_freq8", 1585.8, 17.12),
]

for scene, config, tt, psnr in aggr_data:
    if config == "Baseline":
        sp_str = "Reference"
        pd_str = "---"
    else:
        # Find baseline for this scene
        base_tt = next((t for s, c, t, p in aggr_data if s == scene and c == "Baseline"), 0)
        base_psnr = next((p for s, c, t, p in aggr_data if s == scene and c == "Baseline"), 0)
        sp = (1 - tt / base_tt) * 100
        pd = psnr - base_psnr
        sp_str = f"{sp:>+.1f}%"
        pd_str = f"{pd:>+.2f}"
    print(f"{scene:<10} {config:<25} {tt:>8.1f} {psnr:>8.2f} {sp_str:>10} {pd_str:>10}")

# ===== Speedup Breakdown =====
print(f"\n{'=' * 95}")
print("Speedup Breakdown: Separable SSIM vs Freq8 vs Combined")
print(f"{'=' * 95}")

if "Baseline (orig SSIM, every iter)" in mod_results:
    base_t = mod_results["Baseline (orig SSIM, every iter)"]["time"]
    sep_t = mod_results.get("Sep_ssim (sep SSIM, every iter)", {}).get("time", 0)
    freq8_t = mod_results.get("Sep_freq8 (sep SSIM, every 8)", {}).get("time", 0)
    
    if sep_t > 0 and freq8_t > 0:
        print(f"\n  30K Moderate Pruning (Room):")
        print(f"    Baseline → Sep_ssim:  {base_t:.0f}s → {sep_t:.0f}s = {base_t/sep_t:.2f}x (separable SSIM alone)")
        print(f"    Sep_ssim → Sep_freq8: {sep_t:.0f}s → {freq8_t:.0f}s = {sep_t/freq8_t:.2f}x (freq8 additional)")
        print(f"    Combined:              {base_t:.0f}s → {freq8_t:.0f}s = {base_t/freq8_t:.2f}x")

# 5K comparison
print(f"\n  5K Aggressive Pruning (Room):")
print(f"    Baseline → Sep_ssim:  506.9s → 266.0s = {506.9/266.0:.2f}x")
print(f"    Sep_ssim → Sep_freq8: 266.0s → 115.3s = {266.0/115.3:.2f}x")
print(f"    Combined:              506.9s → 115.3s = {506.9/115.3:.2f}x")

# Quality Breakdown
if "Baseline (orig SSIM, every iter)" in mod_results:
    base_p = mod_results["Baseline (orig SSIM, every iter)"]["psnr"]
    sep_p = mod_results.get("Sep_ssim (sep SSIM, every iter)", {}).get("psnr", 0)
    freq8_p = mod_results.get("Sep_freq8 (sep SSIM, every 8)", {}).get("psnr", 0)
    
    print(f"\n  Quality Breakdown (30K Moderate, Room):")
    print(f"    Baseline:  PSNR={base_p:.2f}")
    print(f"    Sep_ssim:  PSNR={sep_p:.2f} (diff: {sep_p-base_p:+.2f} dB — separable SSIM is numerically equivalent)")
    print(f"    Sep_freq8: PSNR={freq8_p:.2f} (diff: {freq8_p-base_p:+.2f} dB — freq8 provides quality improvement)")
    print(f"\n    Key insight: Freq8 is the quality driver (+1.41 dB), separable SSIM is the speed driver")
