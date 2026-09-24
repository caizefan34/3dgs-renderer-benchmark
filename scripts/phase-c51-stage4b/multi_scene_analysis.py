#!/usr/bin/env python3
"""Phase C51 Stage 4B — Multi-scene analysis."""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4b")

scenes = ["room", "bicycle", "garden"]
methods = ["baseline", "k50_b1"]

results = {}
for scene in scenes:
    for method in methods:
        f = result_dir / f"training_{scene}_{method}.json"
        if f.exists():
            results[f"{scene}_{method}"] = json.load(open(f))

print("=" * 100)
print("Phase C51 Stage 4B — Multi-Scene 30K Results")
print("=" * 100)

# Table A — Main Performance
print("\n### Table A — Main Performance\n")
print(f"| Scene | Method | PSNR | SSIM | Time (ms) | E2E Speedup | Final GS |")
print(f"|-------|--------|------|------|-----------|-------------|----------|")

for scene in scenes:
    b = results.get(f"{scene}_baseline")
    k = results.get(f"{scene}_k50_b1")
    if b:
        print(f"| {scene} | baseline | {b['final_psnr']:.2f} | {b['final_ssim']:.4f} | {b['timing']['mean_ms']:.2f} | — | {b['final_gaussians']:,} |")
    if k and b:
        speedup = (b['timing']['mean_ms'] / k['timing']['mean_ms'] - 1) * 100
        d_psnr = k['final_psnr'] - b['final_psnr']
        d_ssim = k['final_ssim'] - b['final_ssim']
        print(f"| {scene} | K50-B1 | {k['final_psnr']:.2f} | {k['final_ssim']:.4f} | {k['timing']['mean_ms']:.2f} | +{speedup:.1f}% | {k['final_gaussians']:,} |")

# Table B — Population
print("\n### Table B — Population\n")
print(f"| Scene | Method | Clone | Split | Prune | Final GS | GS Ratio |")
print(f"|-------|--------|-------|-------|-------|----------|----------|")

for scene in scenes:
    b = results.get(f"{scene}_baseline")
    k = results.get(f"{scene}_k50_b1")
    if b:
        print(f"| {scene} | baseline | {b['total_clone']:,} | {b['total_split']:,} | {b['total_prune']:,} | {b['final_gaussians']:,} | 100.0% |")
    if k and b:
        gs_ratio = k['final_gaussians'] / b['final_gaussians']
        clone_pct = k['total_clone'] / b['total_clone'] * 100 if b['total_clone'] > 0 else 0
        split_pct = k['total_split'] / b['total_split'] * 100 if b['total_split'] > 0 else 0
        prune_pct = k['total_prune'] / b['total_prune'] * 100 if b['total_prune'] > 0 else 0
        print(f"| {scene} | K50-B1 | {k['total_clone']:,} | {k['total_split']:,} | {k['total_prune']:,} | {k['final_gaussians']:,} | {gs_ratio:.1%} |")

# Gate evaluation
print("\n### Gate Evaluation\n")
print(f"| Scene | ΔPSNR | ΔSSIM | Speedup | Gate B | Gate C | All Pass |")
print(f"|-------|-------|-------|---------|--------|--------|-----------|")

all_pass_count = 0
for scene in scenes:
    b = results.get(f"{scene}_baseline")
    k = results.get(f"{scene}_k50_b1")
    if not b or not k:
        continue
    d_psnr = k['final_psnr'] - b['final_psnr']
    d_ssim = k['final_ssim'] - b['final_ssim']
    speedup = (b['timing']['mean_ms'] / k['timing']['mean_ms'] - 1) * 100
    gate_b = d_psnr >= -0.2 and abs(d_ssim) < 0.005
    gate_c = speedup > 5.0
    all_pass = gate_b and gate_c
    if all_pass:
        all_pass_count += 1
    print(f"| {scene} | {d_psnr:+.2f} | {d_ssim:+.4f} | +{speedup:.1f}% | {'✅' if gate_b else '❌'} | {'✅' if gate_c else '❌'} | {'✅' if all_pass else '❌'} |")

print(f"\nScenes passing all gates: {all_pass_count}/{len(scenes)}")

# Speedup detail
print("\n### Speedup Detail\n")
print(f"| Scene | Baseline ms | K50-B1 ms | Δ ms | Speedup | Fwd+Bwd Δ | Opt Δ | Mask cost |")
print(f"|-------|-------------|-----------|------|---------|-----------|-------|-----------|")

for scene in scenes:
    b = results.get(f"{scene}_baseline")
    k = results.get(f"{scene}_k50_b1")
    if not b or not k:
        continue
    bt = b['timing']
    kt = k['timing']
    delta = bt['mean_ms'] - kt['mean_ms']
    speedup = delta / bt['mean_ms'] * 100
    fwd_bwd_delta = bt['fwd_bwd_mean_ms'] - kt['fwd_bwd_mean_ms']
    opt_delta = bt['opt_mean_ms'] - kt['opt_mean_ms']
    mask_cost = kt['mask_mean_ms']
    print(f"| {scene} | {bt['mean_ms']:.2f} | {kt['mean_ms']:.2f} | {delta:.2f} | +{speedup:.1f}% | {fwd_bwd_delta:.2f} | {opt_delta:.2f} | {mask_cost:.3f} |")

# PSNR trajectory comparison
print("\n### PSNR Trajectory (K50-B1 vs Baseline)\n")
milestones = [0, 1000, 5000, 10000, 15000, 20000, 25000, 30000]

print(f"| Iter |", end="")
for scene in scenes:
    print(f" {scene}_base | {scene}_k50 |", end="")
print()
print(f"|------|", end="")
for scene in scenes:
    print(f"-----------|-----------|", end="")
print()

for m in milestones:
    print(f"| {m} |", end="")
    for scene in scenes:
        for method in ["baseline", "k50_b1"]:
            r = results.get(f"{scene}_{method}")
            if r:
                traj = {t["iter"]: t for t in r.get("trajectory", [])}
                if m in traj:
                    print(f" {traj[m]['psnr']:9.2f} |", end="")
                else:
                    closest = min(traj.keys(), key=lambda k: abs(k - m)) if traj else 0
                    print(f" {traj.get(closest, {}).get('psnr', 0):9.2f} |", end="")
            else:
                print(f"       N/A |", end="")
    print()

# Save analysis
analysis = {}
for scene in scenes:
    b = results.get(f"{scene}_baseline")
    k = results.get(f"{scene}_k50_b1")
    if b and k:
        analysis[scene] = {
            "baseline_psnr": b['final_psnr'],
            "k50_b1_psnr": k['final_psnr'],
            "psnr_delta": k['final_psnr'] - b['final_psnr'],
            "baseline_ssim": b['final_ssim'],
            "k50_b1_ssim": k['final_ssim'],
            "ssim_delta": k['final_ssim'] - b['final_ssim'],
            "baseline_time_ms": b['timing']['mean_ms'],
            "k50_b1_time_ms": k['timing']['mean_ms'],
            "speedup_pct": (b['timing']['mean_ms'] / k['timing']['mean_ms'] - 1) * 100,
            "baseline_gs": b['final_gaussians'],
            "k50_b1_gs": k['final_gaussians'],
            "gs_ratio": k['final_gaussians'] / b['final_gaussians'],
            "clone_ratio": k['total_clone'] / b['total_clone'] if b['total_clone'] > 0 else 0,
            "split_ratio": k['total_split'] / b['total_split'] if b['total_split'] > 0 else 0,
        }

out_file = result_dir / "multi_scene_analysis.json"
with open(out_file, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nAnalysis saved to {out_file}")
