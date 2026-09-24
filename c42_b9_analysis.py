#!/usr/bin/env python3
"""
C42-B9 Final Analysis: Generate all JSON deliverables and decision gate.

Computes:
  - bicycle_refinement.json (B9-D skip decision)
  - room_candidate.json (B9-E: global validation at candidate scale 1.0)
  - garden_candidate.json (B9-E: global validation at candidate scale 1.0)
  - fixed_scale_pareto.json (B9-I)
  - strong_fixed_baseline.json (B9-F)
  - adaptive_vs_strong_fixed.json (B9-G)
  - final_gate.json (B9-H)
  - provenance.json
"""
import json, os, sys
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/c42_adaptive/b9")

# ============================================================
# DATA: Unified 30K metrics (all cameras, same pipeline)
# ============================================================
# From completion batch unified eval
UNIFIED = {
    "room": {
        "1.0":  {"psnr": 31.9414, "ssim": 0.9261, "lpips": 0.3003},
        "0.75": {"psnr": 31.7747, "ssim": 0.9177, "lpips": 0.3169},
        "0.5":  {"psnr": 32.0738, "ssim": 0.9235, "lpips": 0.3075},
    },
    "garden": {
        "1.0":   {"psnr": 29.2113, "ssim": 0.8882, "lpips": 0.1473},
        "0.75":  {"psnr": 28.9116, "ssim": 0.8708, "lpips": 0.1680},
        "0.625": {"psnr": 28.8445, "ssim": 0.8654, "lpips": 0.1746},
        "0.5":   {"psnr": 28.8635, "ssim": 0.8669, "lpips": 0.1721},
    },
    "bicycle": {
        "1.0":   {"psnr": 26.3773, "ssim": 0.8360, "lpips": 0.2463},
        "0.75":  {"psnr": 26.1592, "ssim": 0.8135, "lpips": 0.2694},
        "0.625": {"psnr": 26.1134, "ssim": 0.8052, "lpips": 0.2791},
        "0.5":   {"psnr": 26.3374, "ssim": 0.8133, "lpips": 0.2694},
    },
}

# B9 new Bicycle metrics (from bicycle_080/085/090/095.json final_eval)
B9_BICYCLE = {
    "0.80": {"psnr": 26.1402, "ssim": 0.8144, "lpips": 0.2693},
    "0.85": {"psnr": 26.1556, "ssim": 0.8124, "lpips": 0.2723},
    "0.90": {"psnr": 26.1883, "ssim": 0.8154, "lpips": 0.2695},
    "0.95": {"psnr": 26.0883, "ssim": 0.8155, "lpips": 0.2699},
}

# Merge into unified (B9_BICYCLE keys already 2-decimal)
for k, v in B9_BICYCLE.items():
    UNIFIED["bicycle"][k] = v

# ============================================================
# DATA: Loss-cost curve (B9-A, SepSSIM, A100, 1080p, backward only)
# ============================================================
LOSS_COST = {
    "0.50": 2.42,
    "0.625": 3.54,
    "0.75": 4.92,
    "0.80": 5.51,
    "0.85": 6.24,
    "0.90": 7.00,
    "0.95": 7.77,
    "1.00": 8.34,
}

# ============================================================
# CONSTRAINTS
# ============================================================
TAU = 0.020  # primary SSIM constraint
# Multi-metric: dPSNR >= -0.50 AND abs(dSSIM) <= 0.020 AND dLPIPS <= +0.03

SCENES = ["room", "garden", "bicycle"]
# Use consistent 2-decimal keys; normalize UNIFIED keys to match
ALL_SCALES = ["0.50", "0.625", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"]

def _norm_key(k):
    """Normalize scale key to 2-decimal string."""
    return f"{float(k):.2f}"

# Normalize UNIFIED keys
for scene in UNIFIED:
    new_dict = {}
    for k, v in UNIFIED[scene].items():
        new_dict[_norm_key(k)] = v
    UNIFIED[scene] = new_dict

def get_baseline_ssim(scene):
    return UNIFIED[scene]["1.00"]["ssim"]

def get_baseline_psnr(scene):
    return UNIFIED[scene]["1.00"]["psnr"]

def get_baseline_lpips(scene):
    return UNIFIED[scene]["1.00"]["lpips"]

def dssim(scene, scale):
    """abs(SSIM_1.0 - SSIM_scale)"""
    return abs(get_baseline_ssim(scene) - UNIFIED[scene][scale]["ssim"])

def dpsnr(scene, scale):
    """PSNR_scale - PSNR_1.0"""
    return UNIFIED[scene][scale]["psnr"] - get_baseline_psnr(scene)

def dlpips(scene, scale):
    """LPIPS_scale - LPIPS_1.0 (lower is better)"""
    return UNIFIED[scene][scale]["lpips"] - get_baseline_lpips(scene)

def passes_ssim(scene, scale, tau=TAU):
    return dssim(scene, scale) <= tau

def passes_multimetric(scene, scale, tau=TAU):
    return (dpsnr(scene, scale) >= -0.50 and
            dssim(scene, scale) <= tau and
            dlpips(scene, scale) <= 0.03)

def feasible_scales(scene, tau=TAU, multimetric=False):
    """Return sorted list of feasible scales for a scene."""
    check = passes_multimetric if multimetric else passes_ssim
    return [s for s in ALL_SCALES if s in UNIFIED[scene] and check(scene, s, tau)]

def cost(scale):
    return LOSS_COST[scale]


# ============================================================
# B9-C: Bicycle boundary analysis
# ============================================================
print("=" * 72)
print("B9-C: Bicycle Boundary Analysis")
print("=" * 72)

bike_baseline_ssim = get_baseline_ssim("bicycle")
print(f"\nBicycle 1.0 baseline SSIM = {bike_baseline_ssim:.4f}")
print(f"\n{'Scale':>6s} {'SSIM':>8s} {'dSSIM':>8s} {'Pass?':>6s} {'Cost(ms)':>9s}")
print("-" * 42)
for s in ALL_SCALES:
    if s in UNIFIED["bicycle"]:
        ssim_val = UNIFIED["bicycle"][s]["ssim"]
        ds = abs(bike_baseline_ssim - ssim_val)
        p = "PASS" if ds <= TAU else "FAIL"
        c = cost(s) if s in LOSS_COST else "N/A"
        print(f"{s:>6s} {ssim_val:>8.4f} {ds:>8.4f} {p:>6s} {c:>9}")

bike_feasible = feasible_scales("bicycle")
print(f"\nBicycle feasible set (tau={TAU}): {bike_feasible}")
bike_min_feasible = bike_feasible[0] if bike_feasible else None
print(f"Bicycle minimum tested feasible scale: {bike_min_feasible}")

# B9-D: Skip decision
print("\n" + "=" * 72)
print("B9-D: Boundary Refinement Decision")
print("=" * 72)
print(f"\n  0.90 dSSIM = {dssim('bicycle', '0.90'):.4f} (margin: {dssim('bicycle', '0.90') - TAU:+.4f})")
print(f"  0.95 dSSIM = {dssim('bicycle', '0.95'):.4f} (margin: {dssim('bicycle', '0.95') - TAU:+.4f})")
print(f"\n  Both 0.90 and 0.95 are marginal failures (dSSIM 0.0206, 0.0205 vs tau=0.020).")
print(f"  SSIM does not monotonically improve with scale (0.85 dSSIM=0.0236 > 0.80 dSSIM=0.0216),")
print(f"  indicating training variance dominates at these margins.")
print(f"  Decision: SKIP boundary refinement. Boundary = 1.0.")

bicycle_refinement = {
    "experiment": "C42-B9 Phase D: Boundary Refinement",
    "decision": "SKIP",
    "reason": "Both 0.90 (dSSIM=0.0206) and 0.95 (dSSIM=0.0205) are marginal failures. SSIM is non-monotonic at these scales (0.85 dSSIM=0.0236 > 0.80 dSSIM=0.0216), indicating training variance dominates the <0.001 margins. Testing 0.975 would not yield a conclusive result.",
    "bicycle_dssim_table": {s: {"dssim": dssim("bicycle", s), "pass": passes_ssim("bicycle", s)} for s in ALL_SCALES if s in UNIFIED["bicycle"]},
    "boundary_scale": "1.00",
    "tau": TAU,
}

# ============================================================
# B9-E: Global validation
# ============================================================
print("\n" + "=" * 72)
print("B9-E: Global Validation")
print("=" * 72)

all_feasible = {}
for scene in SCENES:
    fs = feasible_scales(scene)
    all_feasible[scene] = fs
    print(f"  {scene}: feasible = {fs}")

global_intersection = sorted(set.intersection(*[set(all_feasible[s]) for s in SCENES]),
                             key=lambda x: float(x))
print(f"\n  Global intersection (tau={TAU}): {global_intersection}")
global_min = global_intersection[0] if global_intersection else None
print(f"  Minimum globally feasible scale: {global_min}")

# Room and Garden at candidate scale (1.0) — already baselines
room_candidate = {
    "experiment": "C42-B9 Phase E: Global Validation — Room at candidate scale",
    "candidate_scale": global_min,
    "scene": "room",
    "baseline_1.0": UNIFIED["room"]["1.00"],
    "candidate_metrics": UNIFIED["room"][global_min] if global_min else None,
    "dssim_at_candidate": dssim("room", global_min) if global_min else None,
    "passes_ssim": passes_ssim("room", global_min) if global_min else False,
    "passes_multimetric": passes_multimetric("room", global_min) if global_min else False,
    "note": "Candidate scale = 1.0 (only globally feasible scale). Room 1.0 is the baseline, trivially feasible.",
}

garden_candidate = {
    "experiment": "C42-B9 Phase E: Global Validation — Garden at candidate scale",
    "candidate_scale": global_min,
    "scene": "garden",
    "baseline_1.0": UNIFIED["garden"]["1.00"],
    "candidate_metrics": UNIFIED["garden"][global_min] if global_min else None,
    "dssim_at_candidate": dssim("garden", global_min) if global_min else None,
    "passes_ssim": passes_ssim("garden", global_min) if global_min else False,
    "passes_multimetric": passes_multimetric("garden", global_min) if global_min else False,
    "note": "Candidate scale = 1.0 (only globally feasible scale). Garden 1.0 is the baseline, trivially feasible.",
}

# ============================================================
# B9-F: Strongest tested fixed baseline
# ============================================================
print("\n" + "=" * 72)
print("B9-F: Strongest Tested Fixed Baseline")
print("=" * 72)

# Strong fixed baseline = cheapest globally feasible scale
strong_fixed_scale = global_min
strong_fixed_cost = cost(strong_fixed_scale) if strong_fixed_scale else None
print(f"  Strong fixed scale: {strong_fixed_scale}")
print(f"  Cost: {strong_fixed_cost:.2f}ms")

# Check multi-metric at this scale for all scenes
print(f"\n  Multi-metric validation at scale={strong_fixed_scale}:")
for scene in SCENES:
    dp = dpsnr(scene, strong_fixed_scale)
    ds = dssim(scene, strong_fixed_scale)
    dl = dlpips(scene, strong_fixed_scale)
    pm = passes_multimetric(scene, strong_fixed_scale)
    print(f"    {scene}: dPSNR={dp:+.4f} dSSIM={ds:.4f} dLPIPS={dl:+.4f} → {'PASS' if pm else 'FAIL'}")

strong_fixed_baseline = {
    "experiment": "C42-B9 Phase F: Strongest Tested Fixed Baseline",
    "definition": "Cheapest globally feasible fixed scale (satisfies tau=0.020 for all scenes)",
    "global_feasible_set": global_intersection,
    "strong_fixed_scale": strong_fixed_scale,
    "strong_fixed_cost_ms": strong_fixed_cost,
    "per_scene_validation": {
        scene: {
            "scale": strong_fixed_scale,
            "dpsnr": dpsnr(scene, strong_fixed_scale),
            "dssim": dssim(scene, strong_fixed_scale),
            "dlpips": dlpips(scene, strong_fixed_scale),
            "passes_multimetric": passes_multimetric(scene, strong_fixed_scale),
        }
        for scene in SCENES
    },
    "note": "Bicycle is the binding constraint — only 1.0 is feasible for Bicycle at tau=0.020.",
}

# ============================================================
# B9-G: Adaptive advantage vs strong fixed baseline
# ============================================================
print("\n" + "=" * 72)
print("B9-G: Adaptive Oracle vs Strong Fixed Baseline")
print("=" * 72)

# Adaptive oracle: cheapest feasible scale per scene
oracle_selections = {}
for scene in SCENES:
    fs = feasible_scales(scene)
    if fs:
        cheapest = min(fs, key=cost)
        oracle_selections[scene] = {
            "scale": cheapest,
            "cost_ms": cost(cheapest),
            "dssim": dssim(scene, cheapest),
            "passes_multimetric": passes_multimetric(scene, cheapest),
        }
    else:
        oracle_selections[scene] = None

oracle_costs = [oracle_selections[s]["cost_ms"] for s in SCENES]
oracle_avg = sum(oracle_costs) / len(oracle_costs)

print(f"\n  Adaptive oracle selections (tau={TAU}):")
for scene in SCENES:
    sel = oracle_selections[scene]
    print(f"    {scene}: scale={sel['scale']} cost={sel['cost_ms']:.2f}ms dSSIM={sel['dssim']:.4f} "
          f"multimetric={'PASS' if sel['passes_multimetric'] else 'FAIL'}")
print(f"\n  Adaptive oracle average cost: {oracle_avg:.2f}ms")
print(f"  Strong fixed baseline cost: {strong_fixed_cost:.2f}ms")
advantage = strong_fixed_cost - oracle_avg
print(f"  Adaptive advantage: {advantage:.2f}ms")

adaptive_vs_strong_fixed = {
    "experiment": "C42-B9 Phase G: Adaptive Oracle vs Strong Fixed Baseline",
    "tau": TAU,
    "adaptive_oracle": {
        "selections": oracle_selections,
        "per_scene_costs": oracle_costs,
        "average_cost_ms": oracle_avg,
    },
    "strong_fixed_baseline": {
        "scale": strong_fixed_scale,
        "cost_ms": strong_fixed_cost,
    },
    "adaptive_advantage_ms": advantage,
    "all_oracle_selections_pass_multimetric": all(oracle_selections[s]["passes_multimetric"] for s in SCENES),
    "note": "All oracle-selected configurations at the primary tau=0.020 operating point satisfy the LPIPS +0.03 constraint.",
}

# ============================================================
# B9-H: Decision gate
# ============================================================
print("\n" + "=" * 72)
print("B9-H: Decision Gate")
print("=" * 72)

if advantage >= 5.0:
    decision = "KEEP_FOR_PREDICTOR"
elif advantage >= 2.0:
    decision = "MODIFY"
else:
    decision = "DROP_AS_MAIN_METHOD"

print(f"  Adaptive advantage: {advantage:.2f}ms")
print(f"  Decision: {decision}")
print(f"  Rationale: ", end="")
if decision == "KEEP_FOR_PREDICTOR":
    print("Advantage >= 5.0ms. The adaptive oracle provides substantial cost savings over any fixed scale.")
elif decision == "MODIFY":
    print("Advantage 2.0-5.0ms. The adaptive oracle provides moderate savings. Consider modifications to increase the margin (e.g., relaxed tau, additional scenes, or predictor integration).")
else:
    print("Advantage < 2.0ms. The adaptive oracle does not provide sufficient savings over a fixed scale.")

final_gate = {
    "experiment": "C42-B9 Phase H: Decision Gate",
    "tau": TAU,
    "adaptive_advantage_ms": advantage,
    "decision": decision,
    "decision_thresholds": {
        "keep": "advantage >= 5.0ms",
        "modify": "2.0ms <= advantage < 5.0ms",
        "drop": "advantage < 2.0ms",
    },
    "rationale": (
        f"Adaptive advantage = {advantage:.2f}ms (strong fixed = {strong_fixed_cost:.2f}ms at scale {strong_fixed_scale}, "
        f"adaptive oracle avg = {oracle_avg:.2f}ms). "
        f"Bicycle is the binding constraint: only scale 1.0 satisfies tau=0.020 for Bicycle, "
        f"forcing the global fixed scale to 1.0. "
        f"The adaptive oracle saves cost on Room (→0.5) and Garden (→0.75) but must use 1.0 for Bicycle. "
        f"Advantage falls in the MODIFY range."
    ),
    "key_finding": (
        "Among the tested scales {0.5, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0}, only 1.0 was globally feasible at tau=0.020. "
        "Bicycle is the binding constraint: dSSIM at 0.90=0.0206 and 0.95=0.0205, both marginal failures. "
        "The adaptive oracle exploits Room's tolerance (cheapest feasible=0.5) and Garden's tolerance (cheapest feasible=0.75), "
        "but Bicycle forces full-resolution SSIM."
    ),
}

# ============================================================
# B9-I: Fixed-scale Pareto table
# ============================================================
print("\n" + "=" * 72)
print("B9-I: Fixed-Scale Pareto Table")
print("=" * 72)

pareto_entries = []
for s in ALL_SCALES:
    if s not in LOSS_COST:
        continue
    entry = {"scale": s, "loss_cost_ms": cost(s)}
    for scene in SCENES:
        if s in UNIFIED[scene]:
            entry[f"{scene}_dssim"] = dssim(scene, s)
            entry[f"{scene}_passes"] = passes_ssim(scene, s)
    globally_feasible = all(s in UNIFIED[scene] and passes_ssim(scene, s) for scene in SCENES)
    entry["globally_feasible"] = globally_feasible
    pareto_entries.append(entry)
    gf = "GLOBAL" if globally_feasible else ""
    r = "✓" if entry.get("room_passes") else "✗"
    g = "✓" if entry.get("garden_passes") else "✗"
    b = "✓" if entry.get("bicycle_passes") else "✗"
    print(f"  scale={s:>5s}  cost={cost(s):>5.2f}ms  Room={r} Garden={g} Bicycle={b}  {gf}")

fixed_scale_pareto = {
    "experiment": "C42-B9 Phase I: Fixed-Scale Pareto Table",
    "tau": TAU,
    "scales": pareto_entries,
    "globally_feasible_scales": global_intersection,
    "note": "Pareto frontier: only scale 1.0 is globally feasible. Bicycle is the binding constraint at every scale < 1.0.",
}

# ============================================================
# Provenance
# ============================================================
provenance = {
    "experiment": "C42-B9 Strong Fixed-Scale Boundary Search",
    "gaussian_model_path": "baseline/reference_v1/gaussian_model.py",
    "gaussian_model_hash": "68731e375013a6f2",
    "config_path": "baseline/reference_v1/config.py",
    "config_hash": "ca76d4f36059839e",
    "config_base": "REFERENCE_V1_ABSGRAD",
    "loss_definition": "(1-lambda)*L1 + lambda*d_ssim_downsampled(scale), lambda=0.2",
    "ssim_implementation": "SepSSIM window=11 sigma=1.5 C1=(0.01)^2 C2=(0.03)^2",
    "interpolation": "F.interpolate(mode='area')",
    "constraint_tau": TAU,
    "constraint_multimetric": "dPSNR >= -0.50 AND abs(dSSIM) <= 0.020 AND dLPIPS <= +0.03",
    "scenes": SCENES,
    "tested_scales": ALL_SCALES,
    "new_scales_tested_in_b9": ["0.80", "0.85", "0.90", "0.95"],
    "loss_cost_protocol": "A100, 1080p, frozen tensors, SepSSIM, F.interpolate(mode='area'), 50 warmups, 300 timed, CUDA events",
    "provenance_guard": "Startup assertion in c42_b9_train.py and c42_b9_loss_cost.py verifying GaussianModel hash == 68731e375013a6f2",
}

# ============================================================
# Save all JSON files
# ============================================================
files = {
    "provenance.json": provenance,
    "bicycle_refinement.json": bicycle_refinement,
    "room_candidate.json": room_candidate,
    "garden_candidate.json": garden_candidate,
    "fixed_scale_pareto.json": fixed_scale_pareto,
    "strong_fixed_baseline.json": strong_fixed_baseline,
    "adaptive_vs_strong_fixed.json": adaptive_vs_strong_fixed,
    "final_gate.json": final_gate,
}

print("\n" + "=" * 72)
print("Saving JSON deliverables")
print("=" * 72)
for name, data in files.items():
    path = OUTPUT_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {path}")

print(f"\nAll {len(files)} JSON deliverables saved.")
print(f"\nFinal decision: {decision} (advantage = {advantage:.2f}ms)")
