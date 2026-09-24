#!/usr/bin/env python3
"""
Phase C52 Stage 0 — Final Comparison
The most important comparison table across all conditions and scenes.
"""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c52-stage0")

scenes = ["room"]
budgets = [75, 50]

def load(scene, config_name):
    f = result_dir / f"{config_name}_{scene}.json"
    if f.exists():
        return json.load(open(f))
    return None

print("=" * 110)
print("Phase C52 Stage 0 — Final Comparison")
print("=" * 110)

# The most important comparison
print("\n### THE KEY COMPARISON: Predictive vs Uniform vs Oracle at Matched Budget\n")
print(f"| Scene | Budget | Strategy | PSNR | SSIM | Final GS | Clone | Split | Prune | Mean ms |")
print(f"|-------|--------|----------|------|------|----------|-------|-------|-------|---------|")

for scene in scenes:
    # Baseline
    b = load(scene, "baseline") or load(scene, "baseline_30k")
    if b:
        print(f"| {scene} | 100% | baseline | {b['final_psnr']:.2f} | {b['final_ssim']:.4f} | "
              f"{b['final_gaussians']:>8,} | {b['total_clone']:>6,} | {b['total_split']:>6,} | "
              f"{b['total_prune']:>6,} | {b['timing']['mean_ms']:.2f} |")

    for budget in budgets:
        suffix = f"_{budget}_30k" if budget in [75, 50] else ""
        # Also try without _30k suffix for 5K results
        for sfx in [suffix, f"_{budget}"]:
            u = load(scene, f"uniform{sfx}")
            p = load(scene, f"predictive{sfx}")
            o = load(scene, f"oracle{sfx}")
            if u and p:
                print(f"| {scene} | {budget}% | uniform | {u['final_psnr']:.2f} | {u['final_ssim']:.4f} | "
                      f"{u['final_gaussians']:>8,} | {u['total_clone']:>6,} | {u['total_split']:>6,} | "
                      f"{u['total_prune']:>6,} | {u['timing']['mean_ms']:.2f} |")
                print(f"| {scene} | {budget}% | predictive | {p['final_psnr']:.2f} | {p['final_ssim']:.4f} | "
                      f"{p['final_gaussians']:>8,} | {p['total_clone']:>6,} | {p['total_split']:>6,} | "
                      f"{p['total_prune']:>6,} | {p['timing']['mean_ms']:.2f} |")
                if o:
                    print(f"| {scene} | {budget}% | oracle | {o['final_psnr']:.2f} | {o['final_ssim']:.4f} | "
                          f"{o['final_gaussians']:>8,} | {o['total_clone']:>6,} | {o['total_split']:>6,} | "
                          f"{o['total_prune']:>6,} | {o['timing']['mean_ms']:.2f} |")
                break

# Matched-budget comparison
print("\n### Matched-Budget Quality Comparison\n")
print(f"| Scene | Budget | Uniform PSNR | Predictive PSNR | Oracle PSNR | P-U Δ | O-U Δ | P-O Gap |")
print(f"|-------|--------|-------------|-----------------|-------------|-------|-------|---------|")

for scene in scenes:
    for budget in budgets:
        for sfx in [f"_{budget}_30k", f"_{budget}"]:
            u = load(scene, f"uniform{sfx}")
            p = load(scene, f"predictive{sfx}")
            o = load(scene, f"oracle{sfx}")
            if u and p:
                u_psnr = u["final_psnr"]
                p_psnr = p["final_psnr"]
                o_psnr = o["final_psnr"] if o else 0
                pu = p_psnr - u_psnr
                ou = o_psnr - u_psnr if o else 0
                po = p_psnr - o_psnr if o else 0
                print(f"| {scene} | {budget}% | {u_psnr:>11.2f} | {p_psnr:>15.2f} | {o_psnr:>11.2f} | "
                      f"{pu:>+5.2f} | {ou:>+5.2f} | {po:>+7.2f} |")
                break

# Population comparison
print("\n### Population Comparison\n")
print(f"| Scene | Budget | Uniform GS | Predictive GS | Oracle GS | Baseline GS | U/B Ratio | P/B Ratio |")
print(f"|-------|--------|-----------|---------------|-----------|-------------|-----------|-----------|")

for scene in scenes:
    b = load(scene, "baseline") or load(scene, "baseline_30k")
    b_gs = b["final_gaussians"] if b else 0
    for budget in budgets:
        for sfx in [f"_{budget}_30k", f"_{budget}"]:
            u = load(scene, f"uniform{sfx}")
            p = load(scene, f"predictive{sfx}")
            o = load(scene, f"oracle{sfx}")
            if u and p:
                u_gs = u["final_gaussians"]
                p_gs = p["final_gaussians"]
                o_gs = o["final_gaussians"] if o else 0
                u_br = u_gs / b_gs * 100 if b_gs > 0 else 0
                p_br = p_gs / b_gs * 100 if b_gs > 0 else 0
                print(f"| {scene} | {budget}% | {u_gs:>9,} | {p_gs:>13,} | {o_gs:>9,} | {b_gs:>11,} | "
                      f"{u_br:>9.1f}% | {p_br:>9.1f}% |")
                break

# Save final comparison
final = {}
for scene in scenes:
    b = load(scene, "baseline") or load(scene, "baseline_30k")
    if b:
        final[scene] = {"baseline": {
            "psnr": b["final_psnr"], "ssim": b["final_ssim"],
            "gs": b["final_gaussians"], "clone": b["total_clone"],
            "split": b["total_split"], "prune": b["total_prune"],
        }}
        for budget in budgets:
            for sfx in [f"_{budget}_30k", f"_{budget}"]:
                u = load(scene, f"uniform{sfx}")
                p = load(scene, f"predictive{sfx}")
                o = load(scene, f"oracle{sfx}")
                if u and p:
                    final[scene][f"{budget}%"] = {
                        "uniform": {"psnr": u["final_psnr"], "ssim": u["final_ssim"],
                                    "gs": u["final_gaussians"]},
                        "predictive": {"psnr": p["final_psnr"], "ssim": p["final_ssim"],
                                       "gs": p["final_gaussians"]},
                        "p_minus_u": p["final_psnr"] - u["final_psnr"],
                    }
                    if o:
                        final[scene][f"{budget}%"]["oracle"] = {
                            "psnr": o["final_psnr"], "ssim": o["final_ssim"],
                            "gs": o["final_gaussians"]}
                        final[scene][f"{budget}%"]["o_minus_u"] = o["final_psnr"] - u["final_psnr"]
                    break

out_file = result_dir / "final_comparison.json"
with open(out_file, 'w') as f:
    json.dump(final, f, indent=2)
print(f"\nFinal comparison saved to {out_file}")

# Budget Pareto
pareto = {}
for scene in scenes:
    b = load(scene, "baseline") or load(scene, "baseline_30k")
    if b:
        pareto[scene] = {
            "baseline": {"psnr": b["final_psnr"], "gs": b["final_gaussians"]},
        }
        for budget in budgets:
            for sfx in [f"_{budget}_30k", f"_{budget}"]:
                u = load(scene, f"uniform{sfx}")
                p = load(scene, f"predictive{sfx}")
                if u and p:
                    pareto[scene][f"{budget}%"] = {
                        "uniform": {"psnr": u["final_psnr"], "gs": u["final_gaussians"]},
                        "predictive": {"psnr": p["final_psnr"], "gs": p["final_gaussians"]},
                    }
                    break

out_file2 = result_dir / "budget_pareto.json"
with open(out_file2, 'w') as f:
    json.dump(pareto, f, indent=2)
print(f"Budget Pareto saved to {out_file2}")
