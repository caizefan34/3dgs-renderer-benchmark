#!/usr/bin/env python3
"""
Phase C51 Stage 5 — Population Analysis
Compare Gaussian population trajectories across conditions A/B/C/D.
"""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage5")

conditions = {
    "A_baseline": "baseline_room.json",
    "B_k50_b1": "k50_b1_room.json",
    "C_k50_full_dens": "k50_full_dens_room.json",
    "D_k50_masked_opt": "k50_masked_opt_room.json",
}

results = {}
for name, fname in conditions.items():
    f = result_dir / fname
    if f.exists():
        results[name] = json.load(open(f))
    else:
        print(f"WARNING: {fname} not found")

print("=" * 110)
print("Phase C51 Stage 5 — Population Analysis (Room)")
print("=" * 110)

# Population trajectory
milestones = [0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]

print("\n### Gaussian Population Trajectory\n")
header = "| Iter |"
for name in conditions:
    header += f" {name} |"
print(header)
print("|------|" + "-----------|" * len(conditions))

for m in milestones:
    row = f"| {m} |"
    for name in conditions:
        r = results.get(name)
        if r:
            traj = {t["iter"]: t for t in r.get("trajectory", [])}
            if m in traj:
                row += f" {traj[m]['gaussians']:>9,} |"
            elif traj:
                closest = min(traj.keys(), key=lambda k: abs(k - m))
                row += f" {traj[closest]['gaussians']:>9,} |"
            else:
                row += f"       N/A |"
        else:
            row += f"       N/A |"
    print(row)

# Final population comparison
print("\n### Final Population\n")
print(f"| Condition | Final GS | Clone | Split | Prune |")
print(f"|-----------|----------|-------|-------|-------|")
for name in conditions:
    r = results.get(name)
    if r:
        print(f"| {name} | {r['final_gaussians']:>8,} | {r['total_clone']:>6,} | {r['total_split']:>6,} | {r['total_prune']:>6,} |")

# Population ratios (relative to baseline)
print("\n### Population Ratios (relative to A baseline)\n")
baseline = results.get("A_baseline")
if baseline:
    print(f"| Condition | Final GS Ratio | Clone Ratio | Split Ratio | Prune Ratio |")
    print(f"|-----------|----------------|-------------|-------------|-------------|")
    for name in conditions:
        r = results.get(name)
        if r and baseline:
            gs_r = r['final_gaussians'] / baseline['final_gaussians']
            cl_r = r['total_clone'] / baseline['total_clone'] if baseline['total_clone'] > 0 else 0
            sp_r = r['total_split'] / baseline['total_split'] if baseline['total_split'] > 0 else 0
            pr_r = r['total_prune'] / baseline['total_prune'] if baseline['total_prune'] > 0 else 0
            print(f"| {name} | {gs_r:>14.1%} | {cl_r:>11.1%} | {sp_r:>11.1%} | {pr_r:>11.1%} |")

# Densification events comparison
print("\n### Densification Events (first 10 events)\n")
for name in conditions:
    r = results.get(name)
    if r and r.get("densification_events"):
        print(f"\n{name}:")
        print(f"  {'Iter':>6} {'Clone':>8} {'Split':>8} {'Prune':>8} {'GS':>10}")
        for ev in r["densification_events"][:10]:
            print(f"  {ev['iter']:>6} {ev['cloned']:>8} {ev['split']:>8} {ev['pruned']:>8} {ev['gaussians']:>10,}")

# Save analysis
analysis = {}
for name in conditions:
    r = results.get(name)
    if r:
        analysis[name] = {
            "final_gaussians": r["final_gaussians"],
            "total_clone": r["total_clone"],
            "total_split": r["total_split"],
            "total_prune": r["total_prune"],
            "trajectory": [(t["iter"], t["gaussians"]) for t in r.get("trajectory", [])],
            "densification_events": r.get("densification_events", []),
        }

if baseline:
    for name in conditions:
        if name in analysis and "A_baseline" in analysis:
            b = analysis["A_baseline"]
            a = analysis[name]
            a["gs_ratio"] = a["final_gaussians"] / b["final_gaussians"]
            a["clone_ratio"] = a["total_clone"] / b["total_clone"] if b["total_clone"] > 0 else 0
            a["split_ratio"] = a["total_split"] / b["total_split"] if b["total_split"] > 0 else 0
            a["prune_ratio"] = a["total_prune"] / b["total_prune"] if b["total_prune"] > 0 else 0

out_file = result_dir / "population_analysis.json"
with open(out_file, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nAnalysis saved to {out_file}")
