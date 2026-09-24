#!/usr/bin/env python3
"""Phase C51 Stage 4B — Speed attribution and mask cost analysis."""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4b")

# Load 30K results
configs = ["baseline", "k50_b1", "k50_postdens"]
results = {}
for name in configs:
    f = result_dir / f"training_room_{name}.json"
    if f.exists():
        results[name] = json.load(open(f))

if not results:
    print("No results found")
    exit()

print("=" * 80)
print("Speed Attribution Analysis (30K Canonical ROOM)")
print("=" * 80)

# Component breakdown
print("\n--- Timing Decomposition (mean ms per iteration) ---\n")
print(f"{'Component':<15}", end="")
for name in configs:
    if name in results:
        print(f" {name:>12}", end="")
print(f" {'Δ (B1-base)':>12}")
print("-" * 65)

components = ["fwd_bwd_mean_ms", "dens_mean_ms", "opt_mean_ms", "mask_mean_ms"]
component_labels = ["Fwd+Bwd", "Densify", "Optimizer", "Mask"]

baseline_total = 0
k50_total = 0
for comp, label in zip(components, component_labels):
    print(f"{label:<15}", end="")
    vals = {}
    for name in configs:
        if name in results:
            v = results[name]["timing"].get(comp, 0)
            vals[name] = v
            print(f" {v:12.3f}", end="")
    if "baseline" in vals and "k50_b1" in vals:
        delta = vals["k50_b1"] - vals["baseline"]
        print(f" {delta:+12.3f}")
    else:
        print()

# Total
print(f"{'Total':<15}", end="")
for name in configs:
    if name in results:
        v = results[name]["timing"]["mean_ms"]
        print(f" {v:12.3f}", end="")
if "baseline" in results and "k50_b1" in results:
    delta = results["k50_b1"]["timing"]["mean_ms"] - results["baseline"]["timing"]["mean_ms"]
    print(f" {delta:+12.3f}")
else:
    print()

# Speedup attribution
print("\n--- Speedup Attribution ---\n")
if "baseline" in results and "k50_b1" in results:
    b = results["baseline"]["timing"]
    k = results["k50_b1"]["timing"]
    total_saved = b["mean_ms"] - k["mean_ms"]
    fwd_bwd_saved = b["fwd_bwd_mean_ms"] - k["fwd_bwd_mean_ms"]
    opt_saved = b["opt_mean_ms"] - k["opt_mean_ms"]
    mask_cost = k["mask_mean_ms"]  # mask cost is additional
    
    print(f"Total time saved:     {total_saved:.3f} ms/iter ({total_saved/b['mean_ms']*100:.1f}% of baseline)")
    print(f"  Fwd+Bwd saved:      {fwd_bwd_saved:.3f} ms ({fwd_bwd_saved/total_saved*100:.1f}% of saved)")
    print(f"  Optimizer saved:    {opt_saved:.3f} ms ({opt_saved/total_saved*100:.1f}% of saved)")
    print(f"  Mask cost:          -{mask_cost:.3f} ms ({mask_cost/total_saved*100:.1f}% of saved)")
    print(f"  Other:              {total_saved - fwd_bwd_saved - opt_saved + mask_cost:.3f} ms")
    
    print(f"\nMask cost / iteration:    {mask_cost:.3f} ms")
    print(f"Mask cost / total time:   {mask_cost/k['mean_ms']*100:.2f}%")
    print(f"Mask cost / saved time:   {mask_cost/total_saved*100:.1f}%")
    print(f"Net benefit:              {total_saved:.3f} ms/iter ({total_saved/b['mean_ms']*100:.1f}% speedup)")

# Sparse vs dense detailed
print("\n--- Sparse vs Dense Iteration Timing ---\n")
for name in configs:
    if name not in results:
        continue
    t = results[name]["timing"]
    print(f"{name}:")
    print(f"  Sparse: {t['sparse_mean_ms']:.2f}ms ({t['n_sparse_iters']} iters)")
    print(f"  Dense:  {t['dense_mean_ms']:.2f}ms ({t['n_dense_iters']} iters)")
    if t['n_sparse_iters'] > 0:
        delta = t['dense_mean_ms'] - t['sparse_mean_ms']
        print(f"  Δ:      {delta:.2f}ms ({delta/t['dense_mean_ms']*100:.1f}% sparse savings)")
    print()

# P50/P90
print("--- Latency Distribution ---\n")
print(f"{'Config':<15} {'Mean':>8} {'P50':>8} {'P90':>8} {'Std':>8}")
print("-" * 50)
for name in configs:
    if name not in results:
        continue
    t = results[name]["timing"]
    print(f"{name:<15} {t['mean_ms']:8.2f} {t['p50_ms']:8.2f} {t['p90_ms']:8.2f} {t['std_ms']:8.2f}")

# Densification event summary
print("\n--- Densification Summary ---\n")
print(f"{'Config':<15} {'Clone':>10} {'Split':>10} {'Prune':>10} {'Final GS':>10} {'GS Ratio':>9}")
print("-" * 70)
for name in configs:
    if name not in results:
        continue
    r = results[name]
    gs_ratio = r["final_gaussians"] / results["baseline"]["final_gaussians"] if "baseline" in results else 1.0
    print(f"{name:<15} {r['total_clone']:10,} {r['total_split']:10,} {r['total_prune']:10,} "
          f"{r['final_gaussians']:10,} {gs_ratio:8.1%}")
