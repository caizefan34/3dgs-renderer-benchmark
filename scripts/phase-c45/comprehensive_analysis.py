#!/usr/bin/env python3
"""Fixed comprehensive analysis that separates 5K and 30K results."""
import json
from pathlib import Path

d = Path("results/a100/phase-c45")

# C44 baseline for comparison (5K iterations)
c44_baseline_5k = {"time_s": 512.4, "psnr": 24.56, "ssim": 0.7992}
c44_freq8_5k = {"time_s": 151.0, "psnr": 25.06, "ssim": 0.8287}

# Known 5K D_sep_freq8 result (was overwritten by 30K, restored from earlier analysis)
d_sep_freq8_5k = {"time_s": 115.3, "psnr": 25.11, "ssim": 0.8291}

print("=" * 100)
print("Phase C45-C48: Results Summary (5K experiments)")
print("=" * 100)
print(f"{'Experiment':<45} {'Time(s)':>8} {'PSNR':>8} {'SSIM':>8} {'Speedup':>10} {'PSNR_d':>8} {'Decision':>10}")
print("-" * 100)

# 5K results (from files that have iters=5000 or are known 5K)
results_5k = {}
for f in sorted(d.glob("*.json")):
    data = json.load(open(f))
    if "eval_points" not in data or not data["eval_points"]:
        continue
    if data.get("iters", 0) != 5000:
        continue
    final = data["eval_points"][-1]
    tt = data.get("total_train_time_s", 0)
    if tt < 1: continue
    
    name = f.stem
    speedup = (c44_baseline_5k["time_s"] - tt) / c44_baseline_5k["time_s"] * 100
    psnr_diff = final["psnr"] - c44_baseline_5k["psnr"]
    
    if speedup > 5 and psnr_diff > -0.5:
        decision = "KEEP"
    elif speedup < 3:
        decision = "DROP"
    elif psnr_diff < -0.5:
        decision = "DROP"
    else:
        decision = "REVIEW"
    
    print(f"{name:<45} {tt:>8.1f} {final['psnr']:>8.2f} {final['ssim']:>8.4f} {speedup:>+9.1f}% {psnr_diff:>+8.2f} {decision:>10}")
    results_5k[name] = {"time_s": tt, "psnr": final["psnr"], "ssim": final["ssim"],
                        "speedup_pct": speedup, "psnr_diff": psnr_diff, "decision": decision}

# Add known 5K D_sep_freq8 (overwritten by 30K)
print(f"{'D_D_sep_freq8_room (5K, restored)':<45} {d_sep_freq8_5k['time_s']:>8.1f} {d_sep_freq8_5k['psnr']:>8.2f} {d_sep_freq8_5k['ssim']:>8.4f} {'+77.5%':>10} {d_sep_freq8_5k['psnr']-c44_baseline_5k['psnr']:>+8.2f} {'KEEP':>10}")

# Reference
print(f"\n{'C44 baseline (5K reference)':<45} {c44_baseline_5k['time_s']:>8.1f} {c44_baseline_5k['psnr']:>8.2f} {c44_baseline_5k['ssim']:>8.4f} {'    ---':>10} {'    ---':>8}")
print(f"{'C44 freq8 (5K reference)':<45} {c44_freq8_5k['time_s']:>8.1f} {c44_freq8_5k['psnr']:>8.2f} {c44_freq8_5k['ssim']:>8.4f} {'+70.5%':>10} {c44_freq8_5k['psnr']-c44_baseline_5k['psnr']:>+8.2f}")

# 30K results
print(f"\n{'=' * 100}")
print("Phase C45-C48: Results Summary (30K experiments)")
print("=" * 100)
print(f"{'Experiment':<45} {'Time(s)':>8} {'PSNR':>8} {'SSIM':>8} {'GS':>10} {'Note':>20}")
print("-" * 100)

for f in sorted(d.glob("*.json")):
    data = json.load(open(f))
    if "eval_points" not in data or not data["eval_points"]:
        continue
    if data.get("iters", 0) != 30000:
        continue
    final = data["eval_points"][-1]
    tt = data.get("total_train_time_s", 0)
    name = f.stem
    
    # Check for degradation
    psnrs = [ep["psnr"] for ep in data["eval_points"]]
    peak_psnr = max(psnrs)
    peak_iter = data["eval_points"][psnrs.index(max(psnrs))]["iter"]
    degradation = peak_psnr - final["psnr"]
    note = f"peak={peak_psnr:.2f}@{peak_iter}" if degradation > 0.5 else "stable"
    
    print(f"{name:<45} {tt:>8.1f} {final['psnr']:>8.2f} {final['ssim']:>8.4f} {final['gaussians']:>10,} {note:>20}")

# Key comparison
print(f"\n{'=' * 100}")
print("Key Comparison: Speed/Quality at 5K iterations")
print("=" * 100)
print(f"  Baseline (orig SSIM, every iter):  {c44_baseline_5k['time_s']:.1f}s  PSNR={c44_baseline_5k['psnr']:.2f}")
print(f"  C44 freq8 (orig SSIM, every 8):    {c44_freq8_5k['time_s']:.1f}s  PSNR={c44_freq8_5k['psnr']:.2f}")
print(f"  C45 sep+freq8 (sep SSIM, every 8): {d_sep_freq8_5k['time_s']:.1f}s  PSNR={d_sep_freq8_5k['psnr']:.2f}")
print(f"  Total speedup: {c44_baseline_5k['time_s']/d_sep_freq8_5k['time_s']:.2f}x  ({(1-d_sep_freq8_5k['time_s']/c44_baseline_5k['time_s'])*100:.1f}% reduction)")
print(f"  vs C44 freq8:  {c44_freq8_5k['time_s']/d_sep_freq8_5k['time_s']:.2f}x additional speedup from separable SSIM")

# Final ranking
print(f"\n{'=' * 100}")
print("Final Ranking (5K, by speedup)")
print("=" * 100)
all_5k = [
    ("D_sep_freq8 (restored)", d_sep_freq8_5k["time_s"], d_sep_freq8_5k["psnr"]),
    ("B1_early_mid_late", 124.2, 25.08),
    ("C2_fft", 125.0, 23.84),
    ("B1_decreasing", 129.0, 25.09),
    ("D_sep_ssim", 266.0, 24.73),
    ("B3_psnr_aware", 234.1, 24.68),
    ("B2_gradient_aware", 259.9, 24.80),
    ("C3_edge", 325.9, 24.60),
    ("D_baseline", 506.9, 24.58),
    ("C1_laplacian", 1021.1, 22.65),
]
all_5k.sort(key=lambda x: x[1])
print(f"{'Rank':>4} {'Method':<30} {'Time(s)':>8} {'PSNR':>8} {'Speedup':>10} {'Quality':>10}")
print("-" * 75)
for i, (name, tt, psnr) in enumerate(all_5k, 1):
    sp = (1 - tt / c44_baseline_5k["time_s"]) * 100
    q = "OK" if psnr >= c44_baseline_5k["psnr"] - 0.5 else "REGRESSED"
    print(f"{i:>4} {name:<30} {tt:>8.1f} {psnr:>8.2f} {sp:>+9.1f}% {q:>10}")
