#!/usr/bin/env python3
"""R5-A: Convergence visualization — PSNR vs iteration, N_GS vs iteration, paired ΔPSNR."""
import json, os, sys
from pathlib import Path
import numpy as np

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, generating text-only visualization")

R4_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")
R5_BASE = Path("/mnt/storage_pool/liaoyuanjun/r5_a")
SCENES = ["train", "truck"]
SEEDS = [0, 1, 2]
EVAL_ITERS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]

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

# Collect all data
all_data = {}
for scene in SCENES:
    for seed in SEEDS:
        for method in ["baseline", "candidate_c"]:
            ckpts = get_checkpoints(scene, seed, method)
            all_data[(scene, seed, method)] = ckpts

output_dir = Path("/mnt/storage_pool/liaoyuanjun/r5_a/figures")
output_dir.mkdir(parents=True, exist_ok=True)

if HAS_MPL:
    # === Figure 1: PSNR vs Iteration (B vs C, all seeds) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, scene in enumerate(SCENES):
        ax = axes[idx]
        colors_b = ['blue', 'steelblue', 'lightblue']
        colors_c = ['red', 'darkorange', 'gold']
        
        for s_idx, seed in enumerate(SEEDS):
            b_ckpt = all_data.get((scene, seed, "baseline"), {})
            c_ckpt = all_data.get((scene, seed, "candidate_c"), {})
            
            b_iters = sorted([int(k) for k in b_ckpt.keys() if "psnr" in b_ckpt[k]])
            c_iters = sorted([int(k) for k in c_ckpt.keys() if "psnr" in c_ckpt[k]])
            
            b_psnrs = [b_ckpt[str(it)]["psnr"] for it in b_iters]
            c_psnrs = [c_ckpt[str(it)]["psnr"] for it in c_iters]
            
            ax.plot(b_iters, b_psnrs, 'o-', color=colors_b[s_idx], alpha=0.7, 
                    label=f'B seed={seed}', markersize=4)
            ax.plot(c_iters, c_psnrs, 's--', color=colors_c[s_idx], alpha=0.7,
                    label=f'C seed={seed}', markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{scene} — PSNR vs Iteration')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "r5-a-psnr-vs-iteration.png", dpi=150)
    print(f"Saved: {output_dir / 'r5-a-psnr-vs-iteration.png'}")
    plt.close()
    
    # === Figure 2: Gaussian Count vs Iteration ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, scene in enumerate(SCENES):
        ax = axes[idx]
        
        for s_idx, seed in enumerate(SEEDS):
            b_ckpt = all_data.get((scene, seed, "baseline"), {})
            c_ckpt = all_data.get((scene, seed, "candidate_c"), {})
            
            b_iters = sorted([int(k) for k in b_ckpt.keys() if "N_gaussians" in b_ckpt[k]])
            c_iters = sorted([int(k) for k in c_ckpt.keys() if "N_gaussians" in c_ckpt[k]])
            
            b_ngs = [b_ckpt[str(it)]["N_gaussians"] for it in b_iters]
            c_ngs = [c_ckpt[str(it)]["N_gaussians"] for it in c_iters]
            
            ax.plot(b_iters, b_ngs, 'o-', color=colors_b[s_idx], alpha=0.7,
                    label=f'B seed={seed}', markersize=4)
            ax.plot(c_iters, c_ngs, 's--', color=colors_c[s_idx], alpha=0.7,
                    label=f'C seed={seed}', markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('N Gaussians')
        ax.set_title(f'{scene} — Gaussian Count vs Iteration')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.axvline(x=15000, color='gray', linestyle=':', alpha=0.5, label='densify_until')
    
    plt.tight_layout()
    plt.savefig(output_dir / "r5-a-ngs-vs-iteration.png", dpi=150)
    print(f"Saved: {output_dir / 'r5-a-ngs-vs-iteration.png'}")
    plt.close()
    
    # === Figure 3: Paired ΔPSNR per Seed ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(6)
    labels = []
    deltas = []
    colors = []
    
    for scene in SCENES:
        for seed in SEEDS:
            b_ckpt = all_data.get((scene, seed, "baseline"), {})
            c_ckpt = all_data.get((scene, seed, "candidate_c"), {})
            
            b_final = b_ckpt.get("30000", {})
            c_final = c_ckpt.get("30000", {})
            
            if b_final.get("psnr") and c_final.get("psnr"):
                d = c_final["psnr"] - b_final["psnr"]
                deltas.append(d)
                labels.append(f"{scene}\nseed={seed}")
                colors.append('green' if d > 0 else 'red')
    
    bars = ax.bar(x_positions[:len(deltas)], deltas, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x_positions[:len(deltas)])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('ΔPSNR (dB) = C − B')
    ax.set_title('R5-A: Paired ΔPSNR per Seed (T&T)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, deltas):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (0.05 if yval > 0 else -0.1),
                f'{val:+.2f}', ha='center', va='bottom' if yval > 0 else 'top', fontsize=9)
    
    # Add gate threshold annotations
    ax.axhline(y=0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "r5-a-paired-delta-psnr.png", dpi=150)
    print(f"Saved: {output_dir / 'r5-a-paired-delta-psnr.png'}")
    plt.close()
    
    print(f"\nAll figures saved to: {output_dir}")
else:
    # Text-only fallback
    print("\n=== PSNR vs Iteration (text table) ===")
    for scene in SCENES:
        print(f"\n--- {scene} ---")
        print(f"{'Iter':<8}", end="")
        for seed in SEEDS:
            print(f"{'B_s'+str(seed):>8} {'C_s'+str(seed):>8} {'Δ':>7}", end="")
        print()
        
        for it in EVAL_ITERS:
            print(f"{it:<8}", end="")
            for seed in SEEDS:
                b = all_data.get((scene, seed, "baseline"), {}).get(str(it), {})
                c = all_data.get((scene, seed, "candidate_c"), {}).get(str(it), {})
                b_p = b.get("psnr", "")
                c_p = c.get("psnr", "")
                d = f"{c_p - b_p:+.2f}" if b_p and c_p else ""
                b_s = f"{b_p:.2f}" if b_p else "—"
                c_s = f"{c_p:.2f}" if c_p else "—"
                print(f"{b_s:>8} {c_s:>8} {d:>7}", end="")
            print()
    
    print("\n=== Paired ΔPSNR bar chart (text) ===")
    print(f"{'Scene+Seed':<15} {'ΔPSNR':>8} {'Bar':>30}")
    print("-" * 55)
    for scene in SCENES:
        for seed in SEEDS:
            b = all_data.get((scene, seed, "baseline"), {}).get("30000", {})
            c = all_data.get((scene, seed, "candidate_c"), {}).get("30000", {})
            if b.get("psnr") and c.get("psnr"):
                d = c["psnr"] - b["psnr"]
                bar_len = int(abs(d) * 10)
                bar = "+" * bar_len if d > 0 else "-" * bar_len
                print(f"{scene} s{seed:<4} {d:>+8.2f} {bar:>30}")
