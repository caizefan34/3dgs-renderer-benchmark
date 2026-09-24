#!/usr/bin/env python3
"""Analyze 30K results and compare with 5K."""
import json
from pathlib import Path

d = Path("results/a100/phase-c45")

# 30K sep_freq8 room
f = d / "D_D_sep_freq8_room.json"
if f.exists():
    data = json.load(open(f))
    print("D30 sep_freq8 room (30K iters):")
    print(f"  Total time: {data['total_train_time_s']:.1f}s")
    print(f"  Eval trajectory:")
    for ep in data["eval_points"]:
        print(f"    iter={ep['iter']:>5}  PSNR={ep['psnr']:>6.2f}  SSIM={ep['ssim']:.4f}  GS={ep['gaussians']:>10,}")
    
    # Compare with 5K
    f5k = d / "D_D_sep_freq8_room.json"
    if f5k.exists():
        data5k = json.load(open(f5k))
        print(f"\n  5K result: {data5k['total_train_time_s']:.1f}s, PSNR={data5k['eval_points'][-1]['psnr']:.2f}")
        print(f"  30K result: {data['total_train_time_s']:.1f}s, PSNR={data['eval_points'][-1]['psnr']:.2f}")
        
    # Check if PSNR degrades
    psnrs = [ep["psnr"] for ep in data["eval_points"]]
    peak_idx = psnrs.index(max(psnrs))
    print(f"\n  Peak PSNR: {max(psnrs):.2f} at iter {data['eval_points'][peak_idx]['iter']}")
    print(f"  Final PSNR: {psnrs[-1]:.2f} at iter {data['eval_points'][-1]['iter']}")
    if psnrs[-1] < max(psnrs) - 0.5:
        print(f"  WARNING: PSNR degradation of {max(psnrs) - psnrs[-1]:.2f} dB from peak to final")
        print(f"  This suggests overfitting or aggressive pruning issues at 30K")
