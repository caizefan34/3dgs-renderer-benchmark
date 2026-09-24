#!/usr/bin/env python3
"""
Phase C51 Stage 5 — Speed Attribution Analysis
Decompose K50-B1 speedup into:
  - Intrinsic sparse-backward speedup (Condition C vs A)
  - Densification-induced speedup (Condition B vs C)
  - Computation-skip benefit (Condition B vs D)
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
print("Phase C51 Stage 5 — Speed Attribution Analysis (Room)")
print("=" * 110)

# Component timing
print("\n### Component Timing\n")
print(f"| Condition | Mean (ms) | Fwd+Bwd (ms) | Dens (ms) | Opt (ms) | Mask (ms) |")
print(f"|-----------|-----------|--------------|-----------|----------|-----------|")
for name in conditions:
    r = results.get(name)
    if r:
        t = r["timing"]
        print(f"| {name} | {t['mean_ms']:>9.2f} | {t['fwd_bwd_mean_ms']:>12.2f} | {t['dens_mean_ms']:>9.2f} | {t['opt_mean_ms']:>8.2f} | {t['mask_mean_ms']:>9.3f} |")

# Speedup decomposition
print("\n### Speedup Decomposition\n")
A = results.get("A_baseline")
B = results.get("B_k50_b1")
C = results.get("C_k50_full_dens")
D = results.get("D_k50_masked_opt")

if A and B and C:
    A_ms = A["timing"]["mean_ms"]
    B_ms = B["timing"]["mean_ms"]
    C_ms = C["timing"]["mean_ms"]

    B_speedup = (A_ms / B_ms - 1) * 100
    C_speedup = (A_ms / C_ms - 1) * 100
    D_speedup = (A_ms / D["timing"]["mean_ms"] - 1) * 100 if D else 0

    densification_speedup = B_speedup - C_speedup  # B gains extra from reduced population
    intrinsic_speedup = C_speedup  # C preserves population, so speedup is pure sparse

    print(f"| Comparison | Speedup | Interpretation |")
    print(f"|------------|---------|----------------|")
    print(f"| B vs A (total K50-B1) | +{B_speedup:.1f}% | Total practical speedup |")
    print(f"| C vs A (intrinsic sparse) | +{C_speedup:.1f}% | Speedup with preserved densification |")
    print(f"| D vs A (masked optimizer) | +{D_speedup:.1f}% | Full backward + zero masked grads |")
    print(f"| B - C (densification-induced) | +{densification_speedup:.1f}% | Speedup from reduced Gaussian population |")

    # Component-level decomposition
    print(f"\n### Component-Level Speedup (C vs A = intrinsic sparse-backward)\n")
    A_fwd_bwd = A["timing"]["fwd_bwd_mean_ms"]
    C_fwd_bwd = C["timing"]["fwd_bwd_mean_ms"]
    A_opt = A["timing"]["opt_mean_ms"]
    C_opt = C["timing"]["opt_mean_ms"]
    A_dens = A["timing"]["dens_mean_ms"]
    C_dens = C["timing"]["dens_mean_ms"]
    C_mask = C["timing"]["mask_mean_ms"]

    fwd_bwd_saving = A_fwd_bwd - C_fwd_bwd
    opt_saving = A_opt - C_opt
    dens_saving = A_dens - C_dens
    mask_cost = C_mask

    total_saving = A_ms - C_ms

    print(f"| Component | A (ms) | C (ms) | Saving (ms) | % of total |")
    print(f"|-----------|--------|--------|-------------|------------|")
    print(f"| Fwd+Bwd | {A_fwd_bwd:.2f} | {C_fwd_bwd:.2f} | {fwd_bwd_saving:.2f} | {fwd_bwd_saving/total_saving*100 if total_saving > 0 else 0:.0f}% |")
    print(f"| Optimizer | {A_opt:.2f} | {C_opt:.2f} | {opt_saving:.2f} | {opt_saving/total_saving*100 if total_saving > 0 else 0:.0f}% |")
    print(f"| Densification | {A_dens:.2f} | {C_dens:.2f} | {dens_saving:.2f} | {dens_saving/total_saving*100 if total_saving > 0 else 0:.0f}% |")
    print(f"| Mask overhead | 0.00 | {C_mask:.3f} | -{C_mask:.3f} | -{C_mask/total_saving*100 if total_saving > 0 else 0:.0f}% |")
    print(f"| Total | {A_ms:.2f} | {C_ms:.2f} | {total_saving:.2f} | 100% |")

    # B vs D: computation skip vs gradient discard
    if D:
        print(f"\n### B vs D: CUDA Skip vs Gradient Discard\n")
        B_fwd_bwd = B["timing"]["fwd_bwd_mean_ms"]
        D_fwd_bwd = D["timing"]["fwd_bwd_mean_ms"]
        D_ms = D["timing"]["mean_ms"]

        print(f"| Metric | B (CUDA skip) | D (full bwd + mask) | Difference |")
        print(f"|--------|---------------|---------------------|------------|")
        print(f"| Mean (ms) | {B_ms:.2f} | {D_ms:.2f} | {D_ms - B_ms:.2f} |")
        print(f"| Fwd+Bwd (ms) | {B_fwd_bwd:.2f} | {D_fwd_bwd:.2f} | {D_fwd_bwd - B_fwd_bwd:.2f} |")
        print(f"| Opt (ms) | {B['timing']['opt_mean_ms']:.2f} | {D['timing']['opt_mean_ms']:.2f} | {D['timing']['opt_mean_ms'] - B['timing']['opt_mean_ms']:.2f} |")
        print(f"| Final GS | {B['final_gaussians']:,} | {D['final_gaussians']:,} | {D['final_gaussians'] - B['final_gaussians']:,} |")

        skip_benefit = D_fwd_bwd - B_fwd_bwd
        print(f"\nCUDA skip benefit (D fwd+bwd - B fwd+bwd): {skip_benefit:.2f} ms")
        print(f"If positive: CUDA skipping saves real computation time.")
        print(f"If negative: overhead of sparse kernel exceeds savings.")

# Save
attribution = {}
if A and B and C:
    attribution["total_speedup_B"] = (A["timing"]["mean_ms"] / B["timing"]["mean_ms"] - 1) * 100
    attribution["intrinsic_speedup_C"] = (A["timing"]["mean_ms"] / C["timing"]["mean_ms"] - 1) * 100
    attribution["densification_induced_speedup"] = attribution["total_speedup_B"] - attribution["intrinsic_speedup_C"]
    if D:
        attribution["masked_opt_speedup_D"] = (A["timing"]["mean_ms"] / D["timing"]["mean_ms"] - 1) * 100
    attribution["component_savings_C"] = {
        "fwd_bwd_ms": A["timing"]["fwd_bwd_mean_ms"] - C["timing"]["fwd_bwd_mean_ms"],
        "opt_ms": A["timing"]["opt_mean_ms"] - C["timing"]["opt_mean_ms"],
        "dens_ms": A["timing"]["dens_mean_ms"] - C["timing"]["dens_mean_ms"],
        "mask_cost_ms": C["timing"]["mask_mean_ms"],
        "total_ms": A["timing"]["mean_ms"] - C["timing"]["mean_ms"],
    }

out_file = result_dir / "speed_attribution.json"
with open(out_file, 'w') as f:
    json.dump(attribution, f, indent=2)
print(f"\nAnalysis saved to {out_file}")
