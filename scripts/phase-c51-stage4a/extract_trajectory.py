#!/usr/bin/env python3
"""Extract PSNR trajectory at key milestones from training JSONs."""
import json
from pathlib import Path

MILESTONES = [0, 500, 1000, 2000, 3000, 4000, 5000]

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4a")
all_traj = {}
for f in sorted(result_dir.glob("training_5k_*.json")):
    name = f.stem.replace("training_5k_", "")
    if name == "analysis":
        continue
    data = json.load(open(f))
    traj = {t["iter"]: t for t in data.get("trajectory", [])}
    all_traj[name] = traj

# Print table
print(f"{'Iter':<6}", end="")
for name in sorted(all_traj.keys()):
    print(f" {name:>12}", end="")
print()
print("-" * (6 + 13 * len(all_traj)))

for m in MILESTONES:
    print(f"{m:<6}", end="")
    for name in sorted(all_traj.keys()):
        traj = all_traj[name]
        if m in traj:
            print(f" {traj[m]['psnr']:12.2f}", end="")
        else:
            # find closest
            closest = min(traj.keys(), key=lambda k: abs(k - m))
            print(f" {traj[closest]['psnr']:12.2f}", end="")
    print()

print()
print("=== SSIM ===")
print(f"{'Iter':<6}", end="")
for name in sorted(all_traj.keys()):
    print(f" {name:>12}", end="")
print()
print("-" * (6 + 13 * len(all_traj)))

for m in MILESTONES:
    print(f"{m:<6}", end="")
    for name in sorted(all_traj.keys()):
        traj = all_traj[name]
        if m in traj:
            print(f" {traj[m]['ssim']:12.4f}", end="")
        else:
            closest = min(traj.keys(), key=lambda k: abs(k - m))
            print(f" {traj[closest]['ssim']:12.4f}", end="")
    print()

print()
print("=== Gaussian Count ===")
print(f"{'Iter':<6}", end="")
for name in sorted(all_traj.keys()):
    print(f" {name:>12}", end="")
print()
print("-" * (6 + 13 * len(all_traj)))

for m in MILESTONES:
    print(f"{m:<6}", end="")
    for name in sorted(all_traj.keys()):
        traj = all_traj[name]
        if m in traj:
            print(f" {traj[m]['gaussians']:12,}", end="")
        else:
            closest = min(traj.keys(), key=lambda k: abs(k - m))
            print(f" {traj[closest]['gaussians']:12,}", end="")
    print()
