#!/usr/bin/env python3
"""Extract 30K trajectory at key milestones."""
import json
from pathlib import Path

MILESTONES = [0, 1000, 5000, 10000, 15000, 20000, 25000, 30000]

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4a")
configs = ["baseline_30k", "k50_b1_30k"]

all_traj = {}
for name in configs:
    f = result_dir / f"training_5k_{name}.json"
    if not f.exists():
        print(f"  {name}: file not found")
        continue
    data = json.load(open(f))
    traj = {t["iter"]: t for t in data.get("trajectory", [])}
    all_traj[name] = traj

# Print PSNR table
print("=== PSNR ===")
print(f"{'Iter':<8}", end="")
for name in configs:
    print(f" {name:>14}", end="")
print()
print("-" * (8 + 15 * len(configs)))

for m in MILESTONES:
    print(f"{m:<8}", end="")
    for name in configs:
        traj = all_traj.get(name, {})
        if m in traj:
            print(f" {traj[m]['psnr']:14.2f}", end="")
        else:
            closest = min(traj.keys(), key=lambda k: abs(k - m)) if traj else 0
            print(f" {traj.get(closest, {}).get('psnr', 0):14.2f}", end="")
    print()

print("\n=== SSIM ===")
print(f"{'Iter':<8}", end="")
for name in configs:
    print(f" {name:>14}", end="")
print()
print("-" * (8 + 15 * len(configs)))

for m in MILESTONES:
    print(f"{m:<8}", end="")
    for name in configs:
        traj = all_traj.get(name, {})
        if m in traj:
            print(f" {traj[m]['ssim']:14.4f}", end="")
        else:
            closest = min(traj.keys(), key=lambda k: abs(k - m)) if traj else 0
            print(f" {traj.get(closest, {}).get('ssim', 0):14.4f}", end="")
    print()

print("\n=== Gaussian Count ===")
print(f"{'Iter':<8}", end="")
for name in configs:
    print(f" {name:>14}", end="")
print()
print("-" * (8 + 15 * len(configs)))

for m in MILESTONES:
    print(f"{m:<8}", end="")
    for name in configs:
        traj = all_traj.get(name, {})
        if m in traj:
            print(f" {traj[m]['gaussians']:14,}", end="")
        else:
            closest = min(traj.keys(), key=lambda k: abs(k - m)) if traj else 0
            print(f" {traj.get(closest, {}).get('gaussians', 0):14,}", end="")
    print()

# Print final stats
print("\n=== Final Stats ===")
for name in configs:
    f = result_dir / f"training_5k_{name}.json"
    data = json.load(open(f))
    print(f"\n{name}:")
    print(f"  Final PSNR: {data['final_psnr']:.2f}")
    print(f"  Final SSIM: {data['final_ssim']:.4f}")
    print(f"  Final GS: {data['final_gaussians']:,}")
    print(f"  Mean time: {data['timing']['mean_ms']:.2f}ms")
    print(f"  Clone: {data['total_clone']}, Split: {data['total_split']}, Prune: {data['total_prune']}")
    print(f"  Sparse mean: {data['timing']['sparse_mean_ms']:.2f}ms")
    print(f"  Dense mean: {data['timing']['dense_mean_ms']:.2f}ms")
