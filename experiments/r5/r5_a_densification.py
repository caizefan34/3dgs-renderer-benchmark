#!/usr/bin/env python3
"""R5-A: Densification diagnostics — extract Gaussian counts at checkpoints."""
import json, os, sys
from pathlib import Path
import numpy as np

R4_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")
R5_BASE = Path("/mnt/storage_pool/liaoyuanjun/r5_a")
SCENES = ["train", "truck"]
SEEDS = [0, 1, 2]

def get_checkpoints(scene, seed, method):
    if seed == 0:
        run_dir = R4_BASE / scene / method
    else:
        run_dir = R5_BASE / f"{scene}_seed{seed}" / method
    
    metrics_path = run_dir / "training_metrics.json"
    if metrics_path.exists():
        d = json.load(open(metrics_path))
        return d.get("checkpoints", {})
    return {}

print("=" * 120)
print("R5-A: Densification Diagnostics — N_GS(t) at checkpoints")
print("=" * 120)

for scene in SCENES:
    print(f"\n{'='*120}")
    print(f"Scene: {scene}")
    print(f"{'='*120}")
    
    # Collect all checkpoint iterations
    all_iters = set()
    data = {}
    for seed in SEEDS:
        for method in ["baseline", "candidate_c"]:
            ckpts = get_checkpoints(scene, seed, method)
            data[(seed, method)] = ckpts
            all_iters.update(int(k) for k in ckpts.keys())
    
    all_iters = sorted(all_iters)
    
    # Print table
    print(f"\n{'Iter':<8}", end="")
    for seed in SEEDS:
        print(f"{'B_s'+str(seed):>10} {'C_s'+str(seed):>10} {'R_GS':>8}", end="")
    print()
    print("-" * (8 + 3 * 30))
    
    for it in all_iters:
        print(f"{it:<8}", end="")
        for seed in SEEDS:
            b = data.get((seed, "baseline"), {}).get(str(it), {})
            c = data.get((seed, "candidate_c"), {}).get(str(it), {})
            b_n = b.get("N_gaussians", "")
            c_n = c.get("N_gaussians", "")
            r_gs = f"{c_n/b_n:.3f}" if b_n and c_n and b_n > 0 else ""
            b_str = f"{b_n}" if b_n else "—"
            c_str = f"{c_n}" if c_n else "—"
            print(f"{b_str:>10} {c_str:>10} {r_gs:>8}", end="")
        print()
    
    # Phase analysis
    print(f"\n--- Phase Analysis ---")
    phases = [(0, 5000, "0-5K (early densification)"), 
              (5000, 15000, "5-15K (active densification)"),
              (15000, 30000, ">15K (post-densification)")]
    
    for lo, hi, label in phases:
        print(f"\n  {label}:")
        for seed in SEEDS:
            b_data = data.get((seed, "baseline"), {})
            c_data = data.get((seed, "candidate_c"), {})
            
            b_vals = [(int(k), v["N_gaussians"]) for k, v in b_data.items() if lo <= int(k) <= hi and "N_gaussians" in v]
            c_vals = [(int(k), v["N_gaussians"]) for k, v in c_data.items() if lo <= int(k) <= hi and "N_gaussians" in v]
            
            if b_vals and c_vals:
                b_start, b_end = b_vals[0][1], b_vals[-1][1]
                c_start, c_end = c_vals[0][1], c_vals[-1][1]
                b_growth = b_end - b_start
                c_growth = c_end - c_start
                r_start = c_start / b_start if b_start > 0 else 0
                r_end = c_end / b_end if b_end > 0 else 0
                
                print(f"    seed={seed}: B {b_start}→{b_end} (Δ{b_growth:+d}), C {c_start}→{c_end} (Δ{c_growth:+d}), R_GS {r_start:.3f}→{r_end:.3f}")
            else:
                print(f"    seed={seed}: insufficient data")

# Summary
print(f"\n{'='*120}")
print("Summary: Mean R_GS by phase")
print(f"{'='*120}")
print(f"\n{'Phase':<30}", end="")
for scene in SCENES:
    print(f"  {scene:>15}", end="")
print()
print("-" * 65)

for lo, hi, label in [(0, 5000, "0-5K"), (5000, 15000, "5-15K"), (15000, 30000, ">15K")]:
    print(f"{label:<30}", end="")
    for scene in SCENES:
        r_vals = []
        for seed in SEEDS:
            b_data = get_checkpoints(scene, seed, "baseline")
            c_data = get_checkpoints(scene, seed, "candidate_c")
            b_vals = [v["N_gaussians"] for k, v in b_data.items() if lo <= int(k) <= hi and "N_gaussians" in v]
            c_vals = [v["N_gaussians"] for k, v in c_data.items() if lo <= int(k) <= hi and "N_gaussians" in v]
            if b_vals and c_vals:
                r_vals.append(c_vals[-1] / b_vals[-1])
        if r_vals:
            print(f"  {np.mean(r_vals):>15.3f}", end="")
        else:
            print(f"  {'N/A':>15}", end="")
    print()
