#!/usr/bin/env python3
"""Analyze 30K garden result and compare degradation patterns."""
import json
from pathlib import Path

d = Path("results/a100/phase-c45")

# 30K sep_freq8 garden (aggressive)
f = d / "D_D_sep_freq8_garden.json"
if f.exists():
    data = json.load(open(f))
    print("D30 sep_freq8 garden (30K, aggressive pruning):")
    print(f"  Total time: {data['total_train_time_s']:.1f}s")
    print(f"  Eval trajectory:")
    for ep in data["eval_points"]:
        print(f"    iter={ep['iter']:>5}  PSNR={ep['psnr']:>6.2f}  SSIM={ep['ssim']:.4f}  GS={ep['gaussians']:>10,}")
    
    psnrs = [ep["psnr"] for ep in data["eval_points"]]
    peak_idx = psnrs.index(max(psnrs))
    print(f"\n  Peak PSNR: {max(psnrs):.2f} at iter {data['eval_points'][peak_idx]['iter']}")
    print(f"  Final PSNR: {psnrs[-1]:.2f}")
    print(f"  Degradation: {max(psnrs) - psnrs[-1]:.2f} dB")

# Also check the 30K room result (already fetched)
f2 = d / "D_D_sep_freq8_room.json"
if f2.exists():
    data2 = json.load(open(f2))
    if data2.get("iters") == 30000:
        print(f"\nD30 sep_freq8 room (30K, aggressive pruning):")
        print(f"  Total time: {data2['total_train_time_s']:.1f}s")
        psnrs2 = [ep["psnr"] for ep in data2["eval_points"]]
        peak_idx2 = psnrs2.index(max(psnrs2))
        print(f"  Peak PSNR: {max(psnrs2):.2f} at iter {data2['eval_points'][peak_idx2]['iter']}")
        print(f"  Final PSNR: {psnrs2[-1]:.2f}")
        print(f"  Degradation: {max(psnrs2) - psnrs2[-1]:.2f} dB")

# Comparison: aggressive vs standard at iter 1000
print(f"\n{'='*70}")
print("Comparison: Aggressive vs Standard pruning at iter 1000")
print(f"{'='*70}")
print(f"  Aggressive (room):  PSNR=30.57  GS=1,392,490  (peak)")
print(f"  Standard (room):    PSNR=25.71  GS=3,047,796  (still converging)")
print(f"\n  Aggressive converges faster but degrades; standard converges slower but stable")
