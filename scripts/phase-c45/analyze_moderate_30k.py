#!/usr/bin/env python3
"""Analyze the completed moderate sep_freq8 30K result."""
import json
from pathlib import Path

d = Path("results/a100/phase-c45")

# Moderate sep_freq8 30K
f = d / "D_D_sep_freq8_room_mod30k.json"
if f.exists():
    data = json.load(open(f))
    print("=" * 80)
    print("Moderate Pruning 30K: Sep_freq8 Result (COMPLETE)")
    print("=" * 80)
    print(f"  Total time: {data['total_train_time_s']:.1f}s")
    print(f"  Iters: {data.get('iters', 'unknown')}")
    print(f"  Prune config: {data.get('prune_config', 'unknown')}")
    print(f"\n  Eval trajectory:")
    for ep in data["eval_points"]:
        print(f"    iter={ep['iter']:>5}  PSNR={ep['psnr']:>6.2f}  SSIM={ep['ssim']:.4f}  GS={ep['gaussians']:>10,}")
    
    psnrs = [ep["psnr"] for ep in data["eval_points"]]
    peak_idx = psnrs.index(max(psnrs))
    print(f"\n  Peak PSNR: {max(psnrs):.2f} at iter {data['eval_points'][peak_idx]['iter']}")
    print(f"  Final PSNR: {psnrs[-1]:.2f}")
    print(f"  Degradation: {max(psnrs) - psnrs[-1]:.2f} dB")

# Aggressive baseline garden 30K
f2 = d / "D_D_baseline_garden_30k.json"
if f2.exists():
    data2 = json.load(open(f2))
    if data2.get("iters") == 30000:
        print(f"\n{'=' * 80}")
        print("Aggressive Pruning 30K: Baseline Garden (COMPLETE)")
        print("=" * 80)
        print(f"  Total time: {data2['total_train_time_s']:.1f}s")
        print(f"  Final PSNR: {data2['eval_points'][-1]['psnr']:.2f}")
        psnrs2 = [ep["psnr"] for ep in data2["eval_points"]]
        print(f"  Peak PSNR: {max(psnrs2):.2f} at iter {data2['eval_points'][psnrs2.index(max(psnrs2))]['iter']}")
        print(f"  Degradation: {max(psnrs2) - psnrs2[-1]:.2f} dB")

# Summary comparison
print(f"\n{'=' * 80}")
print("30K Room: Aggressive vs Moderate Comparison")
print("=" * 80)

# Aggressive results (known)
print(f"\n  Aggressive pruning:")
print(f"    Baseline:  3434.8s  PSNR=23.59  (degradation 6.42 dB)")
print(f"    Sep_freq8:  909.4s  PSNR=24.23  (degradation 6.34 dB)")
print(f"    Speedup:   3.78x    PSNR diff: +0.64 dB")

# Moderate results
if f.exists():
    tt_mod = data['total_train_time_s']
    final_psnr_mod = data['eval_points'][-1]['psnr']
    psnrs_mod = [ep["psnr"] for ep in data["eval_points"]]
    peak_mod = max(psnrs_mod)
    deg_mod = peak_mod - psnrs_mod[-1]
    print(f"\n  Moderate pruning:")
    print(f"    Sep_freq8: {tt_mod:.1f}s  PSNR={final_psnr_mod:.2f}  (degradation {deg_mod:.2f} dB)")
    print(f"    Baseline:  running (iter 11000, PSNR=23.98)")
    print(f"\n  Moderate vs Aggressive sep_freq8:")
    print(f"    Time:      {tt_mod:.1f}s vs 909.4s (moderate is {tt_mod/909.4:.2f}x)")
    print(f"    PSNR:      {final_psnr_mod:.2f} vs 24.23 (moderate is {final_psnr_mod-24.23:+.2f} dB)")
    print(f"    Degr:      {deg_mod:.2f} dB vs 6.34 dB (moderate degrades less)")
