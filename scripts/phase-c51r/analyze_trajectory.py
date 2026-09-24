#!/usr/bin/env python3
"""Analyze C51-R trajectory data — PSNR gap evolution, error accumulation, refresh effect."""
import json, numpy as np
from pathlib import Path

d = Path("results/a100/phase-c51r")

results = {}
for f in d.glob("*.json"):
    if f.stem in ("final_comparison", "quality_analysis"): continue
    data = json.load(open(f))
    results[f.stem] = data

if "baseline" not in results:
    print("ERROR: baseline not found"); exit(1)

baseline_traj = {t["iter"]: t["psnr"] for t in results["baseline"]["trajectory"]}
baseline_gs_traj = {t["iter"]: t["gaussians"] for t in results["baseline"]["trajectory"]}

print("=" * 100)
print("Phase C51-R: Trajectory Analysis")
print("=" * 100)

# 1. PSNR Gap Evolution
print("\n1. PSNR Gap (baseline - config) Over Training")
print(f"{'Iter':>6}", end="")
configs = ["k90", "k80", "refresh50", "refresh100", "refresh200", "refresh500"]
for name in configs:
    if name in results:
        print(f" {name:>12}", end="")
print()
print("-" * (6 + 13 * len([n for n in configs if n in results])))

key_iters = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999]
for it in key_iters:
    print(f"{it:>6}", end="")
    for name in configs:
        if name not in results: continue
        traj = {t["iter"]: t["psnr"] for t in results[name]["trajectory"]}
        if it in baseline_traj and it in traj:
            gap = baseline_traj[it] - traj[it]
            print(f" {gap:>+12.3f}", end="")
        else:
            print(f" {'---':>12}", end="")
    print()

# 2. Error accumulation: Is the gap growing over time?
print("\n2. Error Accumulation Analysis")
print(f"{'Config':<16} {'Gap@1000':>9} {'Gap@2000':>9} {'Gap@3000':>9} {'Gap@4000':>9} {'Gap@5000':>9} {'Trend':>8}")
print("-" * 70)

for name in configs:
    if name not in results: continue
    traj = {t["iter"]: t["psnr"] for t in results[name]["trajectory"]}
    gaps = []
    for it in [1000, 2000, 3000, 4000, 4999]:
        if it in baseline_traj and it in traj:
            gaps.append(baseline_traj[it] - traj[it])
        else:
            gaps.append(None)

    # Determine trend: is gap growing or plateauing?
    valid_gaps = [g for g in gaps if g is not None]
    if len(valid_gaps) >= 3:
        early = np.mean(valid_gaps[:2])
        late = np.mean(valid_gaps[-2:])
        if late > early * 1.3:
            trend = "GROWING"
        elif late < early * 0.7:
            trend = "SHRINKING"
        else:
            trend = "PLATEAU"
    else:
        trend = "N/A"

    gap_strs = [f"{g:>+9.3f}" if g is not None else f"{'---':>9}" for g in gaps]
    print(f"{name:<16} {''.join(gap_strs)} {trend:>8}")

# 3. Refresh effect: Compare refresh iters vs non-refresh
print("\n3. Refresh Effect: PSNR at refresh vs non-refresh iterations")
print(f"{'Config':<16} {'Refresh iters PSNR':>20} {'Non-refresh PSNR':>18} {'Diff':>8}")
print("-" * 65)

for name in ["refresh50", "refresh100", "refresh200", "refresh500"]:
    if name not in results: continue
    traj = results[name]["trajectory"]
    refresh_psnrs = [t["psnr"] for t in traj if t.get("is_refresh")]
    non_refresh_psnrs = [t["psnr"] for t in traj if not t.get("is_refresh") and t.get("is_sparse")]
    if refresh_psnrs and non_refresh_psnrs:
        r_mean = np.mean(refresh_psnrs)
        nr_mean = np.mean(non_refresh_psnrs)
        print(f"{name:<16} {r_mean:>20.2f} {nr_mean:>18.2f} {r_mean-nr_mean:>+8.2f}")
    else:
        print(f"{name:<16} {'N/A':>20} {'N/A':>18} {'N/A':>8}")

# 4. Gaussian count divergence (densification feedback loop)
print("\n4. Gaussian Count Divergence (|G_config - G_baseline|)")
print(f"{'Iter':>6}", end="")
for name in configs:
    if name in results:
        print(f" {name:>12}", end="")
print()
print("-" * (6 + 13 * len([n for n in configs if n in results])))

for it in [0, 1000, 2000, 3000, 4000, 4999]:
    print(f"{it:>6}", end="")
    for name in configs:
        if name not in results: continue
        traj = {t["iter"]: t["gaussians"] for t in results[name]["trajectory"]}
        if it in baseline_gs_traj and it in traj:
            diff = abs(traj[it] - baseline_gs_traj[it])
            print(f" {diff:>12,}", end="")
        else:
            print(f" {'---':>12}", end="")
    print()

# 5. Clone count comparison (densification feedback)
print("\n5. Densification Event Comparison")
print(f"{'Config':<16} {'Clone':>8} {'Split':>8} {'Prune':>7} {'Clone vs base':>14}")
print("-" * 55)
base_clone = results["baseline"]["total_clone"]
for name in ["baseline"] + configs:
    if name not in results: continue
    r = results[name]
    clone_ratio = r["total_clone"] / base_clone if base_clone > 0 else 0
    print(f"{name:<16} {r['total_clone']:>8,} {r['total_split']:>8,} {r['total_prune']:>7,} {clone_ratio:>13.1%}")

# Save
analysis = {
    "psnr_gap_evolution": {},
    "error_accumulation_trend": {},
    "gaussian_divergence": {},
    "densification_events": {},
}
for name in configs:
    if name not in results: continue
    traj = {t["iter"]: t["psnr"] for t in results[name]["trajectory"]}
    gs_traj = {t["iter"]: t["gaussians"] for t in results[name]["trajectory"]}
    analysis["psnr_gap_evolution"][name] = {
        str(it): baseline_traj.get(it, 0) - traj.get(it, 0)
        for it in key_iters if it in baseline_traj and it in traj
    }
    analysis["gaussian_divergence"][name] = {
        str(it): abs(gs_traj.get(it, 0) - baseline_gs_traj.get(it, 0))
        for it in key_iters if it in baseline_gs_traj and it in gs_traj
    }
    analysis["densification_events"][name] = {
        "clone": results[name]["total_clone"],
        "split": results[name]["total_split"],
        "prune": results[name]["total_prune"],
        "clone_ratio_vs_baseline": results[name]["total_clone"] / base_clone if base_clone > 0 else 0,
    }

with open(d / "trajectory_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print(f"\nSaved to {d / 'trajectory_analysis.json'}")
