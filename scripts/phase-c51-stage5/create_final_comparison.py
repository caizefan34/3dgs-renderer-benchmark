#!/usr/bin/env python3
"""
Phase C51 Stage 5 — Final Comparison
The most important comparison table + mechanism comparison across all scenes.
"""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage5")

scenes = ["room", "bicycle", "garden"]

def load(scene, condition):
    """Load result file for a scene/condition combination."""
    mapping = {
        ("room", "baseline"): "baseline_room.json",
        ("room", "k50_b1"): "k50_b1_room.json",
        ("room", "k50_full_dens"): "k50_full_dens_room.json",
        ("room", "k50_masked_opt"): "k50_masked_opt_room.json",
        ("bicycle", "baseline"): "baseline_bicycle.json",
        ("bicycle", "k50_b1"): "k50_b1_bicycle.json",
        ("bicycle", "k50_full_dens"): "k50_full_dens_bicycle.json",
        ("garden", "baseline"): "baseline_garden.json",
        ("garden", "k50_b1"): "k50_b1_garden.json",
        ("garden", "k50_full_dens"): "k50_full_dens_garden.json",
    }
    fname = mapping.get((scene, condition))
    if not fname:
        return None
    f = result_dir / fname
    if f.exists():
        return json.load(open(f))
    return None

print("=" * 120)
print("Phase C51 Stage 5 — Final Comparison")
print("=" * 120)

# The most important comparison table
print("\n### THE KEY COMPARISON: Sparse Backward Effect vs Densification Effect\n")
print(f"| Scene | Method | Final GS | GS Ratio | E2E Speedup | Backward Δ | Opt Δ | Dens Δ | Mask Cost |")
print(f"|-------|--------|----------|----------|-------------|------------|-------|--------|-----------|")

all_data = {}
for scene in scenes:
    scene_data = {}
    for cond in ["baseline", "k50_b1", "k50_full_dens", "k50_masked_opt"]:
        r = load(scene, cond)
        if r:
            scene_data[cond] = r
    all_data[scene] = scene_data

    b = scene_data.get("baseline")
    for cond in ["baseline", "k50_b1", "k50_full_dens"]:
        r = scene_data.get(cond)
        if not r or not b:
            continue
        gs_ratio = r["final_gaussians"] / b["final_gaussians"]
        speedup = (b["timing"]["mean_ms"] / r["timing"]["mean_ms"] - 1) * 100
        bwd_delta = b["timing"]["fwd_bwd_mean_ms"] - r["timing"]["fwd_bwd_mean_ms"]
        opt_delta = b["timing"]["opt_mean_ms"] - r["timing"]["opt_mean_ms"]
        dens_delta = b["timing"]["dens_mean_ms"] - r["timing"]["dens_mean_ms"]
        mask_cost = r["timing"]["mask_mean_ms"]
        label = {"baseline": "A: Baseline", "k50_b1": "B: K50-B1", "k50_full_dens": "C: Sparse+FullDens"}
        print(f"| {scene} | {label[cond]} | {r['final_gaussians']:>8,} | {gs_ratio:>7.1%} | "
              f"{speedup:>+9.1f}% | {bwd_delta:>+10.2f} | {opt_delta:>+5.2f} | {dens_delta:>+6.2f} | {mask_cost:>9.3f} |")

# Speed decomposition table
print("\n### Speed Decomposition: Intrinsic vs Densification-Induced\n")
print(f"| Scene | B Total Speedup | C Intrinsic Speedup | Dens-Induced (B-C) | C Passes 5%? |")
print(f"|-------|-----------------|---------------------|--------------------:|-------------|")

for scene in scenes:
    d = all_data[scene]
    b = d.get("baseline")
    bb = d.get("k50_b1")
    cc = d.get("k50_full_dens")
    if not (b and bb and cc):
        continue
    b_sp = (b["timing"]["mean_ms"] / bb["timing"]["mean_ms"] - 1) * 100
    c_sp = (b["timing"]["mean_ms"] / cc["timing"]["mean_ms"] - 1) * 100
    dens_sp = b_sp - c_sp
    passes = "✅" if c_sp > 5.0 else "❌"
    print(f"| {scene} | +{b_sp:.1f}% | +{c_sp:.1f}% | +{dens_sp:.1f}% | {passes} |")

# Quality comparison
print("\n### Quality Comparison: A vs B vs C\n")
print(f"| Scene | A PSNR | B PSNR | C PSNR | B-A Δ | C-A Δ | C-B Δ | A SSIM | B SSIM | C SSIM |")
print(f"|-------|--------|--------|--------|-------|-------|-------|--------|--------|--------|")

for scene in scenes:
    d = all_data[scene]
    a = d.get("baseline")
    b = d.get("k50_b1")
    c = d.get("k50_full_dens")
    if not a:
        continue
    a_p, b_p, c_p = a["final_psnr"], b["final_psnr"] if b else 0, c["final_psnr"] if c else 0
    a_s, b_s, c_s = a["final_ssim"], b["final_ssim"] if b else 0, c["final_ssim"] if c else 0
    ba = f"{b_p - a_p:+.2f}" if b else "N/A"
    ca = f"{c_p - a_p:+.2f}" if c else "N/A"
    cb = f"{c_p - b_p:+.2f}" if (b and c) else "N/A"
    print(f"| {scene} | {a_p:.2f} | {b_p:.2f} | {c_p:.2f} | {ba} | {ca} | {cb} | "
          f"{a_s:.4f} | {b_s:.4f} | {c_s:.4f} |")

# Population trajectory comparison
print("\n### Population Trajectory (GS at milestones)\n")
milestones = [0, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000]
print(f"| Scene | Iter |", end="")
for cond in ["A", "B", "C"]:
    print(f" {cond} GS |", end="")
print()
print(f"|-------|------|", end="")
for _ in range(3):
    print(f"---------|", end="")
print()

for scene in scenes:
    d = all_data[scene]
    conds = {"A": d.get("baseline"), "B": d.get("k50_b1"), "C": d.get("k50_full_dens")}
    for m in milestones:
        row = f"| {scene} | {m:>5} |"
        all_found = True
        for label in ["A", "B", "C"]:
            r = conds[label]
            if r:
                traj = {t["iter"]: t for t in r.get("trajectory", [])}
                if m in traj:
                    row += f" {traj[m]['gaussians']:>7,} |"
                else:
                    row += f"       ? |"
            else:
                row += f"     N/A |"
                all_found = False
        print(row)

# Save final comparison
final = {}
for scene in scenes:
    d = all_data[scene]
    a, b, c = d.get("baseline"), d.get("k50_b1"), d.get("k50_full_dens")
    dd = d.get("k50_masked_opt")
    if a and b and c:
        final[scene] = {
            "A_baseline": {
                "psnr": a["final_psnr"], "ssim": a["final_ssim"],
                "gs": a["final_gaussians"], "mean_ms": a["timing"]["mean_ms"],
                "fwd_bwd_ms": a["timing"]["fwd_bwd_mean_ms"],
                "opt_ms": a["timing"]["opt_mean_ms"],
            },
            "B_k50_b1": {
                "psnr": b["final_psnr"], "ssim": b["final_ssim"],
                "gs": b["final_gaussians"], "mean_ms": b["timing"]["mean_ms"],
                "fwd_bwd_ms": b["timing"]["fwd_bwd_mean_ms"],
                "opt_ms": b["timing"]["opt_mean_ms"],
                "e2e_speedup": (a["timing"]["mean_ms"] / b["timing"]["mean_ms"] - 1) * 100,
                "gs_ratio": b["final_gaussians"] / a["final_gaussians"],
            },
            "C_k50_full_dens": {
                "psnr": c["final_psnr"], "ssim": c["final_ssim"],
                "gs": c["final_gaussians"], "mean_ms": c["timing"]["mean_ms"],
                "fwd_bwd_ms": c["timing"]["fwd_bwd_mean_ms"],
                "opt_ms": c["timing"]["opt_mean_ms"],
                "e2e_speedup": (a["timing"]["mean_ms"] / c["timing"]["mean_ms"] - 1) * 100,
                "gs_ratio": c["final_gaussians"] / a["final_gaussians"],
            },
            "decomposition": {
                "total_speedup_B": (a["timing"]["mean_ms"] / b["timing"]["mean_ms"] - 1) * 100,
                "intrinsic_speedup_C": (a["timing"]["mean_ms"] / c["timing"]["mean_ms"] - 1) * 100,
                "densification_induced_speedup": ((a["timing"]["mean_ms"] / b["timing"]["mean_ms"] - 1) -
                                                  (a["timing"]["mean_ms"] / c["timing"]["mean_ms"] - 1)) * 100,
            },
        }
        if dd:
            final[scene]["D_k50_masked_opt"] = {
                "psnr": dd["final_psnr"], "ssim": dd["final_ssim"],
                "gs": dd["final_gaussians"], "mean_ms": dd["timing"]["mean_ms"],
                "e2e_speedup": (a["timing"]["mean_ms"] / dd["timing"]["mean_ms"] - 1) * 100,
            }

out_file = result_dir / "final_comparison.json"
with open(out_file, 'w') as f:
    json.dump(final, f, indent=2)
print(f"\nFinal comparison saved to {out_file}")

# Mechanism comparison
mechanism = {}
for scene, d in final.items():
    mechanism[scene] = {
        "total_speedup_pct": d["decomposition"]["total_speedup_B"],
        "intrinsic_sparse_pct": d["decomposition"]["intrinsic_speedup_C"],
        "densification_induced_pct": d["decomposition"]["densification_induced_speedup"],
        "intrinsic_fraction": d["decomposition"]["intrinsic_speedup_C"] / d["decomposition"]["total_speedup_B"]
                              if d["decomposition"]["total_speedup_B"] > 0 else 0,
        "densification_fraction": d["decomposition"]["densification_induced_speedup"] / d["decomposition"]["total_speedup_B"]
                                  if d["decomposition"]["total_speedup_B"] > 0 else 0,
    }

out_file2 = result_dir / "mechanism_comparison.json"
with open(out_file2, 'w') as f:
    json.dump(mechanism, f, indent=2)
print(f"Mechanism comparison saved to {out_file2}")
