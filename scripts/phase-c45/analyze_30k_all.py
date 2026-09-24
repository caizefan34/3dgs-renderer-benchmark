#!/usr/bin/env python3
"""Analyze bicycle 30K result and moderate pruning progress."""
import json
from pathlib import Path

d = Path("results/a100/phase-c45")

# Bicycle 30K aggressive
f = d / "D_D_sep_freq8_bicycle.json"
if f.exists():
    data = json.load(open(f))
    print("D30 sep_freq8 bicycle (30K, aggressive pruning):")
    print(f"  Total time: {data['total_train_time_s']:.1f}s")
    print(f"  Iters: {data.get('iters', 'unknown')}")
    print(f"  Eval trajectory (first/peak/final):")
    eps = data["eval_points"]
    psnrs = [ep["psnr"] for ep in eps]
    peak_idx = psnrs.index(max(psnrs))
    print(f"    First: iter={eps[0]['iter']}  PSNR={eps[0]['psnr']:.2f}  GS={eps[0]['gaussians']:,}")
    print(f"    Peak:  iter={eps[peak_idx]['iter']}  PSNR={eps[peak_idx]['psnr']:.2f}  GS={eps[peak_idx]['gaussians']:,}")
    print(f"    Final: iter={eps[-1]['iter']}  PSNR={eps[-1]['psnr']:.2f}  GS={eps[-1]['gaussians']:,}")
    print(f"  Degradation: {max(psnrs) - psnrs[-1]:.2f} dB")

# Summary of all 30K results so far
print(f"\n{'='*80}")
print("30K Validation Summary (all configs)")
print(f"{'='*80}")
print(f"{'Config':<35} {'Scene':<10} {'Time(s)':>8} {'Peak PSNR':>10} {'Final PSNR':>10} {'Degrad':>8}")
print("-" * 80)

for f in sorted(d.glob("*.json")):
    data = json.load(open(f))
    if data.get("iters") != 30000: continue
    if "eval_points" not in data or not data["eval_points"]: continue
    final = data["eval_points"][-1]
    tt = data.get("total_train_time_s", 0)
    if tt < 1: continue
    psnrs = [ep["psnr"] for ep in data["eval_points"]]
    peak = max(psnrs)
    deg = peak - psnrs[-1]
    name = f.stem
    scene = data.get("scene", "?")
    print(f"{name:<35} {scene:<10} {tt:>8.1f} {peak:>10.2f} {psnrs[-1]:>10.2f} {deg:>8.2f}")

# Moderate pruning progress (not yet complete)
print(f"\n{'='*80}")
print("Moderate 30K Progress (incomplete, latest eval points)")
print(f"{'='*80}")
for f in sorted(d.glob("*mod30k*.json")):
    data = json.load(open(f))
    if "eval_points" not in data or not data["eval_points"]: continue
    if data.get("total_train_time_s", 0) < 1: continue
    final = data["eval_points"][-1]
    name = f.stem
    print(f"  {name}: iter={final['iter']}  PSNR={final['psnr']:.2f}  GS={final['gaussians']:,}")
