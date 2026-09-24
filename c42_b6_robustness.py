#!/usr/bin/env python3
"""C42++ B6 — Oracle Robustness Gate.

B6-A: UNKNOWN-scale sensitivity enumeration
B6-B: Value of information for each missing scale
B6-C: Correct Room statement
B6-D: Multi-metric constrained oracle (PSNR/SSIM/LPIPS)
B6-E: Cost-model robustness (fixed-tensor vs realized scene cost)
B6-F: Counterfactual wording correction

Pure offline CPU analysis. No GPU, no training.
"""
import json, io, itertools
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results" / "c42_adaptive"
RESULTS.mkdir(parents=True, exist_ok=True)

def load(path):
    with io.open(str(path), encoding="utf-8") as f:
        return json.load(f)

def save(path, data):
    with io.open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ============================================================
# CANONICAL DATA (30K)
# ============================================================

# PSNR at 30K (from pareto_summary.json + S2.2 report)
PSNR_30K = {
    "room":    {"1.0": 32.30, "0.75": None,  "0.625": None,  "0.5": 32.54},
    "garden":  {"1.0": 29.63, "0.75": 29.28, "0.625": None,  "0.5": 29.23},
    "bicycle": {"1.0": 26.47, "0.75": None,  "0.625": 26.14, "0.5": 26.43},
}

# SSIM at 30K
SSIM_30K = {
    "room":    {"1.0": 0.9263, "0.75": None,  "0.625": None,  "0.5": 0.9185},
    "garden":  {"1.0": 0.8994, "0.75": 0.8811, "0.625": None,  "0.5": 0.8770},
    "bicycle": {"1.0": 0.8383, "0.75": None,  "0.625": 0.8084, "0.5": 0.8170},
}

# LPIPS at 30K (lower is better; delta = LPIPS_s - LPIPS_1.0 > 0 means degradation)
LPIPS_30K = {
    "room":    {"1.0": None,   "0.75": None,  "0.625": None,  "0.5": None},     # No LPIPS for Room
    "garden":  {"1.0": 0.1400, "0.75": 0.1613, "0.625": None,  "0.5": 0.1655},
    "bicycle": {"1.0": 0.2436, "0.75": None,  "0.625": 0.2750, "0.5": 0.2633},
}

# Loss costs (fixed-tensor, A100, forward+backward, historical)
LOSS_COST = {"1.0": 33.15, "0.75": 19.41, "0.625": 13.92, "0.5": 9.24}

# Total iteration times (scene-specific, includes rasterization, from pareto_summary)
TOTAL_ITER_MS = {
    "room":    {"1.0": 55.4, "0.75": None,  "0.625": None,  "0.5": 29.1},
    "garden":  {"1.0": 67.1, "0.75": 50.4,  "0.625": None,  "0.5": 39.9},
    "bicycle": {"1.0": 75.3, "0.75": None,  "0.625": 50.0,  "0.5": 47.2},
}

# Realized loss forward costs (phase_timing.json, only 1.0 and 0.5)
REALIZED_LOSS_FWD_MS = {
    "garden":  {"1.0": 24.90, "0.5": 7.12},
    "bicycle": {"1.0": 24.98, "0.5": 6.91},
    # Room: no phase_timing
}

# Per-phase total iteration timing (from training JSONs)
PHASE_TIMING = {
    "garden": {
        "1.0": {"0-5K": 60.34, "5-10K": 68.41, "10-15K": 70.12, "15-20K": 68.07, "20-25K": 67.82, "25-30K": 67.61},
        "0.5": {"0-5K": 34.53, "5-10K": 41.10, "10-15K": 42.13, "15-20K": 40.71, "20-25K": 40.53, "25-30K": 40.46},
    },
    "bicycle": {
        "1.0": {"0-5K": None, "5-10K": None, "10-15K": None, "15-20K": None, "20-25K": None, "25-30K": None},  # Need to extract
        "0.5": {"0-5K": None, "5-10K": None, "10-15K": None, "15-20K": None, "20-25K": None, "25-30K": None},
    },
}

SCENES = ["room", "garden", "bicycle"]
SCALES = ["1.0", "0.75", "0.625", "0.5"]
TAUS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

# Missing 30K measurements
UNKNOWN_SCALES = [
    ("room", "0.75"),
    ("room", "0.625"),
    ("garden", "0.625"),
    ("bicycle", "0.75"),
]


def delta(metric_table, scene, scale):
    """Return metric_s - metric_1.0 for scene. None if UNKNOWN."""
    v1 = metric_table[scene]["1.0"]
    vs = metric_table[scene][scale]
    if v1 is None or vs is None:
        return None
    return vs - v1


def abs_delta(metric_table, scene, scale):
    """Return |metric_s - metric_1.0|. None if UNKNOWN."""
    d = delta(metric_table, scene, scale)
    return abs(d) if d is not None else None


# ============================================================
# B6-A: UNKNOWN-scale sensitivity enumeration
# ============================================================

def phase_b6a():
    """Enumerate all UNKNOWN-scale feasible/infeasible assignments."""
    result = {
        "phase": "B6-A",
        "description": "UNKNOWN-scale sensitivity: enumerate all 2^4=16 assignments per tau",
        "unknown_scales": [list(us) for us in UNKNOWN_SCALES],
        "tau_results": {},
    }

    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        tau_result = {"tau": tau, "assignments": [], "min_advantage": None, "max_advantage": None}

        # Enumerate all 2^4 = 16 combinations of feasible/infeasible for UNKNOWN scales
        for bits in itertools.product([True, False], repeat=len(UNKNOWN_SCALES)):
            # bits[i] = True means UNKNOWN_SCALES[i] is feasible
            assignment = {}
            for i, (scene, scale) in enumerate(UNKNOWN_SCALES):
                assignment[f"{scene}_{scale}"] = "feasible" if bits[i] else "infeasible"

            # Build per-scene feasible sets
            scene_feasible = {}
            for scene in SCENES:
                feasible = []
                for s in SCALES:
                    if s == "1.0":
                        feasible.append(s)  # Always feasible
                        continue
                    d = abs_delta(SSIM_30K, scene, s)
                    if d is None:
                        # Check if this is an UNKNOWN scale in our assignment
                        key = f"{scene}_{s}"
                        if key in assignment:
                            if assignment[key] == "feasible":
                                feasible.append(s)
                        # If not in assignment (shouldn't happen), skip
                    else:
                        if d <= tau:
                            feasible.append(s)
                scene_feasible[scene] = feasible

            # Adaptive oracle: cheapest feasible per scene
            oracle_scales = {}
            oracle_costs = {}
            for scene in SCENES:
                fs = scene_feasible[scene]
                if fs:
                    best = min(fs, key=lambda s: LOSS_COST[s])
                    oracle_scales[scene] = best
                    oracle_costs[scene] = LOSS_COST[best]
                else:
                    oracle_scales[scene] = None
                    oracle_costs[scene] = None

            oracle_avg_cost = sum(c for c in oracle_costs.values() if c is not None) / max(1, sum(1 for c in oracle_costs.values() if c is not None))

            # Global feasible: intersection
            global_feasible = set(scene_feasible["room"]) & set(scene_feasible["garden"]) & set(scene_feasible["bicycle"])
            global_feasible_sorted = sorted(global_feasible)

            # Best fixed feasible
            if global_feasible:
                bf_scale = min(global_feasible, key=lambda s: LOSS_COST[s])
                bf_cost = LOSS_COST[bf_scale]
            else:
                bf_scale = None
                bf_cost = None

            # Adaptive advantage
            if bf_cost is not None and oracle_avg_cost is not None:
                advantage = round(bf_cost - oracle_avg_cost, 2)
            else:
                advantage = None

            tau_result["assignments"].append({
                "assignment": assignment,
                "scene_feasible": {s: scene_feasible[s] for s in SCENES},
                "oracle_scales": oracle_scales,
                "oracle_avg_cost_ms": round(oracle_avg_cost, 2) if oracle_avg_cost else None,
                "global_feasible": global_feasible_sorted,
                "best_fixed_feasible_scale": bf_scale,
                "best_fixed_feasible_cost_ms": bf_cost,
                "advantage_ms": advantage,
            })

        # Find min and max advantage
        advantages = [a["advantage_ms"] for a in tau_result["assignments"] if a["advantage_ms"] is not None]
        if advantages:
            tau_result["min_advantage"] = min(advantages)
            tau_result["max_advantage"] = max(advantages)
            tau_result["nominal_advantage"] = [a["advantage_ms"] for a in tau_result["assignments"]
                                                if all(a["assignment"][k] == "infeasible" for k in a["assignment"])][0]

        result["tau_results"][tau_str] = tau_result

    # Summary
    result["summary"] = {}
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        tr = result["tau_results"][tau_str]
        result["summary"][tau_str] = {
            "min_advantage_ms": tr["min_advantage"],
            "max_advantage_ms": tr["max_advantage"],
            "nominal_advantage_ms": tr.get("nominal_advantage"),
            "advantage_range_ms": round(tr["max_advantage"] - tr["min_advantage"], 2) if tr["min_advantage"] is not None and tr["max_advantage"] is not None else None,
            "robust": tr["min_advantage"] is not None and tr["min_advantage"] > 0.5,
        }

    save(RESULTS / "b6a_unknown_sensitivity.json", result)
    print("\n=== B6-A UNKNOWN SENSITIVITY ===")
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        s = result["summary"][tau_str]
        print(f"  tau={tau:.3f}: min={s['min_advantage_ms']}ms, nominal={s['nominal_advantage_ms']}ms, "
              f"max={s['max_advantage_ms']}ms, range={s['advantage_range_ms']}ms, robust={s['robust']}")
    return result


# ============================================================
# B6-B: Value of information for each missing scale
# ============================================================

def phase_b6b(b6a):
    """Rank missing scales by maximum possible change in adaptive advantage."""
    result = {
        "phase": "B6-B",
        "description": "Value of information: max possible change in DC for each missing scale",
        "rankings": [],
    }

    for scene, scale in UNKNOWN_SCALES:
        key = f"{scene}_{scale}"
        max_swing = 0
        best_tau = None
        details_per_tau = {}

        for tau in TAUS:
            tau_str = f"{tau:.3f}"
            assignments = b6a["tau_results"][tau_str]["assignments"]

            # Find max advantage when this scale is feasible
            adv_feasible = [a["advantage_ms"] for a in assignments
                           if a["assignment"].get(key) == "feasible" and a["advantage_ms"] is not None]
            # Find max advantage when this scale is infeasible
            adv_infeasible = [a["advantage_ms"] for a in assignments
                             if a["assignment"].get(key) == "infeasible" and a["advantage_ms"] is not None]

            if adv_feasible and adv_infeasible:
                swing = max(adv_feasible) - max(adv_infeasible)
                swing_abs = abs(swing)
                if swing_abs > max_swing:
                    max_swing = swing_abs
                    best_tau = tau
                details_per_tau[tau_str] = {
                    "max_adv_feasible": max(adv_feasible),
                    "max_adv_infeasible": max(adv_infeasible),
                    "swing": round(swing, 2),
                    "swing_abs": round(swing_abs, 2),
                }

        result["rankings"].append({
            "scale": f"{scene}_{scale}",
            "max_swing_ms": round(max_swing, 2),
            "best_tau": best_tau,
            "details": details_per_tau,
        })

    # Sort by max swing descending
    result["rankings"].sort(key=lambda x: x["max_swing_ms"], reverse=True)

    save(RESULTS / "b6b_value_of_information.json", result)
    print("\n=== B6-B VALUE OF INFORMATION ===")
    for r in result["rankings"]:
        print(f"  {r['scale']}: max_swing={r['max_swing_ms']}ms at tau={r['best_tau']}")
    return result


# ============================================================
# B6-C: Correct Room statement
# ============================================================

def phase_b6c():
    """Document the correct Room statement."""
    result = {
        "phase": "B6-C",
        "description": "Correct Room oracle statement",
        "room_30k_data": {
            "1.0": {"ssim": 0.9263, "psnr": 32.30, "loss_cost_ms": 33.15},
            "0.5": {"ssim": 0.9185, "psnr": 32.54, "loss_cost_ms": 9.24, "abs_delta_ssim": 0.0078},
            "0.75": {"ssim": None, "psnr": None, "loss_cost_ms": 19.41, "status": "UNKNOWN"},
            "0.625": {"ssim": None, "psnr": None, "loss_cost_ms": 13.92, "status": "UNKNOWN"},
        },
        "correction": {
            "tau_ge_0.010": (
                "For tau >= 0.010, Room scale=0.5 is feasible (|dSSIM|=0.0078 <= tau) and is the cheapest "
                "tested candidate (9.24ms). Room 0.75 (19.41ms) and 0.625 (13.92ms) are MORE expensive than 0.5. "
                "Therefore, even if Room 0.75/0.625 are feasible at tau >= 0.010, the Room oracle CANNOT improve "
                "below 9.24ms. These measurements have ZERO value for the Room adaptive oracle at tau >= 0.010."
            ),
            "tau_0.005": (
                "For tau=0.005, Room 0.5 is INFEASIBLE (|dSSIM|=0.0078 > 0.005). If Room 0.75 or 0.625 were feasible, "
                "the Room oracle could be 19.41ms or 13.92ms instead of 33.15ms (scale 1.0). "
                "However, 0.75 has |dSSIM| likely >= 0.0078 (more aggressive downsampling), so feasibility at tau=0.005 is unlikely. "
                "0.625 is even more aggressive. Room 0.75/0.625 MIGHT matter at tau=0.005 but probably won't help."
            ),
            "global_feasibility": (
                "Room 0.75/0.625 may matter for GLOBAL fixed-policy feasibility. "
                "At tau=0.010, global feasible = {1.0} because Room 0.5 is feasible but Garden/Bicycle 0.5 are not. "
                "If Room 0.75 were globally feasible (unlikely since Garden/Bicycle 0.75 fail), it wouldn't change the global set. "
                "Room 0.75/0.625 do NOT affect global feasibility because their feasibility is scene-specific and "
                "global feasibility requires ALL scenes to pass."
            ),
            "conclusion": (
                "Room 0.75 and 0.625 measurements have NEGLIGIBLE value for the adaptive oracle at tau >= 0.010 "
                "(Room 0.5 already optimal) and LOW value at tau=0.005 (unlikely to be feasible). "
                "They do not affect global fixed-policy feasibility. "
                "These measurements should NOT be prioritized for GPU measurement."
            ),
        },
    }
    save(RESULTS / "b6c_room_correction.json", result)
    print("\n=== B6-C ROOM CORRECTION ===")
    print(f"  {result['correction']['tau_ge_0.010'][:120]}...")
    return result


# ============================================================
# B6-D: Multi-metric constrained oracle
# ============================================================

def phase_b6d():
    """Multi-metric oracle: PSNR, SSIM, LPIPS constraints."""

    # Define threshold families
    # Format: (name, constraints)
    # constraints: dict of metric -> threshold
    #   "ssim_abs_delta" -> tau_S (|dSSIM| <= tau_S)
    #   "psnr_delta" -> eps_P (dPSNR >= -eps_P, i.e., PSNR_s >= PSNR_1.0 - eps_P)
    #   "lpips_delta" -> eps_L (dLPIPS <= eps_L, i.e., LPIPS_s <= LPIPS_1.0 + eps_L)

    families = [
        ("SSIM_only_tau010", {"ssim_abs_delta": 0.010}),
        ("SSIM_only_tau020", {"ssim_abs_delta": 0.020}),
        ("PSNR020_SSIM010", {"psnr_delta": 0.20, "ssim_abs_delta": 0.010}),
        ("PSNR050_SSIM010", {"psnr_delta": 0.50, "ssim_abs_delta": 0.010}),
        ("PSNR020_SSIM020", {"psnr_delta": 0.20, "ssim_abs_delta": 0.020}),
        ("PSNR050_SSIM020", {"psnr_delta": 0.50, "ssim_abs_delta": 0.020}),
        ("PSNR050_SSIM010_LPIPS003", {"psnr_delta": 0.50, "ssim_abs_delta": 0.010, "lpips_delta": 0.03}),
        ("PSNR050_SSIM020_LPIPS003", {"psnr_delta": 0.50, "ssim_abs_delta": 0.020, "lpips_delta": 0.03}),
    ]

    result = {
        "phase": "B6-D",
        "description": "Multi-metric constrained oracle: PSNR + SSIM + LPIPS",
        "metric_data_available": {
            "psnr": {s: {sc: PSNR_30K[s][sc] for sc in SCALES} for s in SCENES},
            "ssim": {s: {sc: SSIM_30K[s][sc] for sc in SCALES} for s in SCENES},
            "lpips": {s: {sc: LPIPS_30K[s][sc] for sc in SCALES} for s in SCENES},
            "lpips_note": "Room has NO LPIPS data. LPIPS constraints cannot be evaluated for Room.",
        },
        "families": {},
    }

    for fam_name, constraints in families:
        fam_result = {
            "constraints": constraints,
            "per_scene_feasible": {},
            "per_scene_oracle": {},
            "adaptive_oracle_avg_cost": None,
            "global_feasible": [],
            "best_fixed_feasible": {},
            "advantage_ms": None,
            "unknown_configurations": [],
        }

        oracle_costs = []

        for scene in SCENES:
            feasible = []
            has_unknown = False

            for s in SCALES:
                if s == "1.0":
                    feasible.append(s)
                    continue

                # Check each constraint
                ok = True
                unknown = False

                # SSIM constraint
                if "ssim_abs_delta" in constraints:
                    d = abs_delta(SSIM_30K, scene, s)
                    if d is None:
                        unknown = True
                    elif d > constraints["ssim_abs_delta"]:
                        ok = False

                # PSNR constraint: dPSNR >= -eps_P
                if "psnr_delta" in constraints:
                    d = delta(PSNR_30K, scene, s)
                    if d is None:
                        unknown = True
                    elif d < -constraints["psnr_delta"]:
                        ok = False

                # LPIPS constraint: dLPIPS <= eps_L
                if "lpips_delta" in constraints:
                    d = delta(LPIPS_30K, scene, s)
                    if d is None:
                        unknown = True
                    elif d > constraints["lpips_delta"]:
                        ok = False

                if unknown:
                    has_unknown = True
                    # UNKNOWN scales are excluded from feasible set (conservative)
                    fam_result["unknown_configurations"].append(f"{scene}_{s}")
                elif ok:
                    feasible.append(s)

            fam_result["per_scene_feasible"][scene] = feasible

            # Oracle: cheapest feasible
            if feasible:
                best = min(feasible, key=lambda sc: LOSS_COST[sc])
                fam_result["per_scene_oracle"][scene] = {"scale": best, "cost_ms": LOSS_COST[best]}
                oracle_costs.append(LOSS_COST[best])
            else:
                fam_result["per_scene_oracle"][scene] = {"scale": None, "cost_ms": None}

        # Adaptive oracle average
        if oracle_costs:
            fam_result["adaptive_oracle_avg_cost"] = round(sum(oracle_costs) / len(oracle_costs), 2)

        # Global feasible
        gfs = set(fam_result["per_scene_feasible"]["room"]) & \
              set(fam_result["per_scene_feasible"]["garden"]) & \
              set(fam_result["per_scene_feasible"]["bicycle"])
        fam_result["global_feasible"] = sorted(gfs)

        if gfs:
            bf = min(gfs, key=lambda sc: LOSS_COST[sc])
            fam_result["best_fixed_feasible"] = {"scale": bf, "cost_ms": LOSS_COST[bf]}

            # Advantage
            if fam_result["adaptive_oracle_avg_cost"] is not None:
                fam_result["advantage_ms"] = round(
                    fam_result["best_fixed_feasible"]["cost_ms"] - fam_result["adaptive_oracle_avg_cost"], 2)
                fam_result["advantage_pct"] = round(
                    fam_result["advantage_ms"] / fam_result["best_fixed_feasible"]["cost_ms"] * 100, 1)

        result["families"][fam_name] = fam_result

    # Summary: does opportunity depend on SSIM-only?
    ssim_only_010 = result["families"]["SSIM_only_tau010"]["advantage_ms"]
    psnr_ssim_010 = result["families"]["PSNR050_SSIM010"]["advantage_ms"]
    full_010 = result["families"]["PSNR050_SSIM010_LPIPS003"]["advantage_ms"]
    ssim_only_020 = result["families"]["SSIM_only_tau020"]["advantage_ms"]
    psnr_ssim_020 = result["families"]["PSNR050_SSIM020"]["advantage_ms"]
    full_020 = result["families"]["PSNR050_SSIM020_LPIPS003"]["advantage_ms"]

    result["summary"] = {
        "ssim_only_tau010_advantage": ssim_only_010,
        "psnr_ssim_tau010_advantage": psnr_ssim_010,
        "full_metric_tau010_advantage": full_010,
        "ssim_only_tau020_advantage": ssim_only_020,
        "psnr_ssim_tau020_advantage": psnr_ssim_020,
        "full_metric_tau020_advantage": full_020,
        "opportunity_depends_on_ssim_only": "NO — opportunity exists under PSNR+SSIM constraints too" if psnr_ssim_010 and psnr_ssim_010 > 0.5 else "YES — opportunity disappears under multi-metric",
        "strongest_supported_constraint": "PSNR+SSIM at tau_S=0.020, eps_P=0.50" if psnr_ssim_020 and psnr_ssim_020 > 0.5 else "SSIM-only at tau=0.020",
        "lpips_impact": "LPIPS constraint makes Room UNKNOWN (no LPIPS data), reducing oracle to scale 1.0 for Room. This may reduce or eliminate the advantage.",
    }

    save(RESULTS / "b6d_multi_metric.json", result)
    print("\n=== B6-D MULTI-METRIC ORACLE ===")
    for fam_name, fr in result["families"].items():
        oracle = {s: fr["per_scene_oracle"][s]["scale"] for s in SCENES}
        print(f"  {fam_name}: oracle={oracle}, avg={fr['adaptive_oracle_avg_cost']}ms, "
              f"fixed_feasible={fr['best_fixed_feasible'].get('scale')}, advantage={fr['advantage_ms']}ms")
    print(f"\n  Opportunity depends on SSIM-only? {result['summary']['opportunity_depends_on_ssim_only']}")
    return result


# ============================================================
# B6-E: Cost-model robustness
# ============================================================

def phase_b6e():
    """Distinguish fixed-tensor cost-model estimate from realized scene cost."""

    result = {
        "phase": "B6-E",
        "description": "Cost-model robustness: fixed-tensor vs realized scene cost",
        "fixed_tensor_costs": LOSS_COST,
        "fixed_tensor_source": "results/reference_v1/s23/fixed_tensor_loss_benchmark.json (50 warmup, 300 timed, CUDA events, 1080p frozen tensors, A100)",
        "realized_loss_costs": {},
        "total_iter_times": TOTAL_ITER_MS,
        "phase_timing_available": {},
        "analysis": {},
    }

    # Realized loss forward costs (phase_timing.json)
    result["realized_loss_costs"] = {
        "garden": {"1.0": 24.90, "0.5": 7.12, "0.75": None, "0.625": None},
        "bicycle": {"1.0": 24.98, "0.5": 6.91, "0.75": None, "0.625": None},
        "room": {"1.0": None, "0.5": None, "0.75": None, "0.625": None},
        "source": "results/reference_v1/s22/{garden,bicycle}/phase_timing.json",
        "note": "phase_timing.loss_ms measures loss FORWARD pass during actual training. "
                "Fixed-tensor benchmark measures forward+backward of isolated SSIM. "
                "These are NOT directly comparable — phase_timing loss_ms is forward only, "
                "while fixed-tensor includes backward.",
    }

    # Phase timing total
    result["phase_timing_available"] = {
        "garden": {"1.0": "yes", "0.5": "yes", "0.75": "no", "0.625": "no"},
        "bicycle": {"1.0": "yes", "0.5": "yes", "0.75": "no", "0.625": "no"},
        "room": {"1.0": "no", "0.5": "no", "0.75": "no", "0.625": "no"},
        "note": "Phase timing (detailed forward/loss/backward breakdown) available only for "
                "Garden and Bicycle at scales 1.0 and 0.5. No intermediate-scale phase timing.",
    }

    # Loss timing isolated benchmark (from phase_timing.json loss_timing section)
    result["loss_timing_benchmark"] = {
        "garden": {
            "baseline_full_ssim": {"fwd": 54.79, "bwd": 9.03, "total": 63.82},
            "c42_full_ssim": {"fwd": 24.96, "bwd": 8.19, "total": 33.15},
            "c42_ds05_ssim": {"fwd": 6.92, "bwd": 2.27, "total": 9.18},
        },
        "bicycle": {
            "baseline_full_ssim": {"fwd": 32.19, "bwd": 8.40, "total": 40.60},
            "c42_full_ssim": {"fwd": 24.95, "bwd": 8.18, "total": 33.13},
            "c42_ds05_ssim": {"fwd": 6.91, "bwd": 2.26, "total": 9.17},
        },
        "note": "Isolated SSIM timing benchmark. baseline_full_ssim differs across scenes (54.79 vs 32.19) "
                "because Bicycle baseline used different N_gaussians. c42_full_ssim is consistent (~33.15). "
                "The fixed-tensor benchmark uses same frozen tensors across scenes, hence scene-independent costs.",
    }

    # Analysis
    result["analysis"] = {
        "fixed_tensor_estimate": {
            "max_advantage_ms": 12.55,
            "max_advantage_pct": 37.9,
            "at_tau": 0.020,
            "cost_model": "fixed-tensor forward+backward, scene-independent, 1080p",
            "status": "This is a FIXED-TENSOR COST-MODEL ESTIMATE, not a measured adaptive training saving.",
        },
        "realized_scene_cost_available": "PARTIAL",
        "realized_scene_cost_detail": (
            "Realized loss forward costs are available ONLY for Garden and Bicycle at scales 1.0 and 0.5 "
            "(from phase_timing.json). No realized costs at 0.75 or 0.625. No realized costs for Room. "
            "The realized forward costs differ from the fixed-tensor total costs because: "
            "(1) phase_timing measures forward only, (2) actual training uses variable image sizes, "
            "(3) cache effects during training differ from isolated benchmark. "
            "Therefore the 24-38% opportunity CANNOT be validated as a realized training saving "
            "using existing data."
        ),
        "total_iter_time_analysis": {
            "garden_1.0_vs_0.5": f"{TOTAL_ITER_MS['garden']['1.0']}ms vs {TOTAL_ITER_MS['garden']['0.5']}ms = {TOTAL_ITER_MS['garden']['1.0']/TOTAL_ITER_MS['garden']['0.5']:.2f}x total speedup",
            "bicycle_1.0_vs_0.5": f"{TOTAL_ITER_MS['bicycle']['1.0']}ms vs {TOTAL_ITER_MS['bicycle']['0.5']}ms = {TOTAL_ITER_MS['bicycle']['1.0']/TOTAL_ITER_MS['bicycle']['0.5']:.2f}x total speedup",
            "room_1.0_vs_0.5": f"{TOTAL_ITER_MS['room']['1.0']}ms vs {TOTAL_ITER_MS['room']['0.5']}ms = {TOTAL_ITER_MS['room']['1.0']/TOTAL_ITER_MS['room']['0.5']:.2f}x total speedup",
            "note": "Total iteration time includes rasterization (fwd+bwd) which depends on N_gaussians, not just loss. "
                    "The oracle models LOSS cost only, but total training speed also changes due to different N_gaussians trajectories.",
        },
        "conclusion": (
            "The 24-38% oracle opportunity is a NORMALIZED_FIXED_TENSOR_OPPORTUNITY estimate. "
            "REALIZED_SCENE_COST_OPPORTUNITY cannot be computed from existing data because: "
            "(1) no realized loss costs at intermediate scales (0.75, 0.625), "
            "(2) no phase timing for Room, "
            "(3) the fixed-tensor benchmark is scene-independent while actual loss costs vary by scene. "
            "The realized opportunity may be larger or smaller than the fixed-tensor estimate."
        ),
    }

    save(RESULTS / "b6e_cost_model.json", result)
    print("\n=== B6-E COST MODEL ROBUSTNESS ===")
    print(f"  Fixed-tensor max advantage: {result['analysis']['fixed_tensor_estimate']['max_advantage_ms']}ms ({result['analysis']['fixed_tensor_estimate']['max_advantage_pct']}%)")
    print(f"  Realized scene cost available: {result['analysis']['realized_scene_cost_available']}")
    print(f"  Conclusion: {result['analysis']['conclusion'][:120]}...")
    return result


# ============================================================
# B6-F: Counterfactual wording
# ============================================================

def phase_b6f():
    result = {
        "phase": "B6-F",
        "description": "Counterfactual wording correction",
        "previous_wording": "The offline oracle is an upper bound on online adaptive scheduling.",
        "corrected_wording": (
            "The oracle is a fixed-trajectory counterfactual opportunity proxy. "
            "Because switching supervision scales changes the subsequent optimization trajectory, "
            "it is neither a formal upper nor lower bound on online adaptive training performance."
        ),
        "rationale": (
            "The previous wording claimed the oracle is an 'upper bound', implying online adaptive training "
            "can only do worse. This is incorrect: (1) online scale-switching may discover better optimization "
            "paths than any fixed scale, making it potentially better than the oracle; (2) it may also discover "
            "worse paths. Without running an actual adaptive experiment, neither direction can be established. "
            "The oracle is a PROXY for opportunity, not a BOUND."
        ),
    }
    save(RESULTS / "b6f_counterfactual.json", result)
    print("\n=== B6-F COUNTERFACTUAL WORDING ===")
    print(f"  Corrected: {result['corrected_wording'][:120]}...")
    return result


# ============================================================
# FINAL GATE
# ============================================================

def final_gate(b6a, b6b, b6c, b6d, b6e, b6f):
    """Apply robustness gate."""

    # Key metrics
    # 1. SSIM-only min advantage over all UNKNOWN assignments
    min_adv = None
    max_adv = None
    nominal_adv = None
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        s = b6a["summary"][tau_str]
        if s["min_advantage_ms"] is not None:
            if min_adv is None or s["min_advantage_ms"] < min_adv:
                min_adv = s["min_advantage_ms"]
                min_adv_tau = tau
            if max_adv is None or s["max_advantage_ms"] > max_adv:
                max_adv = s["max_advantage_ms"]
                max_adv_tau = tau
            # Nominal = all UNKNOWN infeasible
            if nominal_adv is None or s["nominal_advantage_ms"] > nominal_adv:
                nominal_adv = s["nominal_advantage_ms"]
                nominal_adv_tau = tau

    # 2. Multi-metric opportunity
    multi_metric_adv = b6d["summary"]["psnr_ssim_tau020_advantage"]
    full_metric_adv = b6d["summary"]["full_metric_tau020_advantage"]

    # 3. Robustness: is min advantage > 0.5ms at the max-opportunity tau?
    max_adv_tau_str = f"{max_adv_tau:.3f}" if max_adv else None
    robust_at_max = False
    if max_adv_tau_str and max_adv_tau_str in b6a["summary"]:
        robust_at_max = b6a["summary"][max_adv_tau_str]["min_advantage_ms"] > 0.5

    # 4. Is the opportunity robust under multi-metric?
    multi_metric_pass = multi_metric_adv is not None and multi_metric_adv > 0.5

    # 5. Highest value missing measurement
    highest_value = b6b["rankings"][0] if b6b["rankings"] else None

    # 6. Cost model
    cost_model = b6e["analysis"]

    gate = {
        "experiment": "C42_ORACLE_ROBUSTNESS",
        "date": "2026-09-15",
        "ssim_only": {
            "min_advantage_ms": min_adv,
            "min_advantage_tau": min_adv_tau,
            "nominal_advantage_ms": nominal_adv,
            "nominal_advantage_tau": nominal_adv_tau,
            "max_advantage_ms": max_adv,
            "max_advantage_tau": max_adv_tau,
            "robust_at_max_tau": robust_at_max,
        },
        "multi_metric": {
            "psnr_ssim_tau020_advantage_ms": multi_metric_adv,
            "full_metric_tau020_advantage_ms": full_metric_adv,
            "full_metric_tau010_advantage_ms": b6d["families"]["PSNR050_SSIM010_LPIPS003"]["advantage_ms"],
            "opportunity_passes_multi_metric": multi_metric_pass,
            "full_metric_survives_at_tau020": full_metric_adv is not None and full_metric_adv > 0.5,
            "lpips_caveat": (
                "LPIPS constraint makes Room UNKNOWN (no LPIPS data at any scale), forcing Room oracle to 1.0. "
                "This reduces advantage from 12.55ms to 4.58ms at tau=0.020 and eliminates it at tau=0.010. "
                "This is a DATA LIMITATION (Room has no LPIPS measurements), not a quality limitation. "
                "If Room LPIPS were available and passed, the full-metric advantage would be larger."
            ),
            "strongest_supported_constraint": b6d["summary"]["strongest_supported_constraint"],
        },
        "unknown_sensitivity": {
            "highest_value_missing_measurement": highest_value["scale"] if highest_value else None,
            "max_effect_on_advantage_ms": highest_value["max_swing_ms"] if highest_value else None,
            "ranking": [r["scale"] for r in b6b["rankings"]],
        },
        "cost_model": {
            "fixed_tensor_estimate_ms": cost_model["fixed_tensor_estimate"]["max_advantage_ms"],
            "fixed_tensor_estimate_pct": cost_model["fixed_tensor_estimate"]["max_advantage_pct"],
            "realized_scene_cost_available": cost_model["realized_scene_cost_available"],
            "conclusion": cost_model["conclusion"],
        },
        "fixed_trajectory_limitation": b6f["corrected_wording"],
        "room_correction": b6c["correction"]["conclusion"],
        "decision": "",
        "rationale": "",
        "predictor_search_justified": False,
        "gpu_measurement_justified": False,
    }

    # Decision logic:
    # PASS: opportunity is robust (min advantage > 0.5ms at max-opportunity tau) AND multi-metric passes
    # MODIFY: opportunity exists but not fully robust (min advantage may be 0 at some UNKNOWN assignments,
    #         or multi-metric reduces but doesn't eliminate it)
    # DROP: opportunity disappears under UNKNOWN sensitivity or multi-metric

    # Check: at the nominal (all UNKNOWN = infeasible) case, is there opportunity?
    nominal_has_opportunity = nominal_adv is not None and nominal_adv > 0.5

    # Check: does the opportunity survive UNKNOWN sensitivity?
    # At the max-opportunity tau (0.020), what's the min advantage?
    min_at_max_tau = b6a["summary"].get("0.020", {}).get("min_advantage_ms")

    # Check: does the opportunity survive multi-metric?
    multi_metric_survives = multi_metric_pass

    if nominal_has_opportunity and min_at_max_tau is not None and min_at_max_tau > 0.5 and multi_metric_survives:
        gate["decision"] = "PASS"
        gate["predictor_search_justified"] = True
        gate["rationale"] = (
            f"Adaptive opportunity is ROBUST: min advantage over all UNKNOWN assignments = {min_at_max_tau}ms at tau=0.020, "
            f"multi-metric (PSNR+SSIM) advantage = {multi_metric_adv}ms. "
            "The opportunity does not depend on SSIM-only constraint and survives UNKNOWN-scale sensitivity. "
            f"CAVEAT: Under full PSNR+SSIM+LPIPS constraint, advantage reduces to {full_metric_adv}ms at tau=0.020 "
            "because Room has no LPIPS data (data limitation, not quality limitation). "
            "The LPIPS-reduced advantage is still material (> 0.5ms threshold). "
            "Predictor search is justified."
        )
    elif nominal_has_opportunity and multi_metric_survives:
        gate["decision"] = "MODIFY"
        gate["predictor_search_justified"] = True
        gate["rationale"] = (
            f"Adaptive opportunity exists nominally ({nominal_adv}ms) and survives multi-metric ({multi_metric_adv}ms), "
            f"but is NOT robust under all UNKNOWN assignments (min advantage = {min_at_max_tau}ms at tau=0.020). "
            "Some UNKNOWN scale measurements could eliminate the advantage. "
            "Predictor search is justified but UNKNOWN scale measurements should be prioritized to confirm robustness."
        )
    elif nominal_has_opportunity:
        gate["decision"] = "MODIFY"
        gate["predictor_search_justified"] = False
        gate["rationale"] = (
            f"Adaptive opportunity exists under SSIM-only ({nominal_adv}ms) but does NOT survive multi-metric constraints "
            f"(PSNR+SSIM advantage = {multi_metric_adv}ms). "
            "The opportunity may depend on choosing SSIM as the only quality constraint. "
            "Predictor search is NOT justified until the multi-metric robustness is established."
        )
    else:
        gate["decision"] = "DROP"
        gate["predictor_search_justified"] = False
        gate["rationale"] = "Adaptive opportunity does not exist even under SSIM-only constraint."

    # GPU measurement justification
    # Only justify if the highest-value measurement could materially change the decision
    if highest_value and highest_value["max_swing_ms"] > 2.0:
        gate["gpu_measurement_justified"] = True
        gate["if_gpu_measurement_justified"] = {
            "single_highest_value_measurement": highest_value["scale"],
            "reason": f"Could change advantage by up to {highest_value['max_swing_ms']}ms",
        }
    else:
        gate["gpu_measurement_justified"] = False
        gate["if_gpu_measurement_justified"] = None

    save(RESULTS / "b6_final_gate.json", gate)
    print("\n" + "=" * 70)
    print("=== B6 FINAL GATE ===")
    print(f"  Decision: {gate['decision']}")
    print(f"  SSIM-only: min={gate['ssim_only']['min_advantage_ms']}ms, nominal={gate['ssim_only']['nominal_advantage_ms']}ms, max={gate['ssim_only']['max_advantage_ms']}ms")
    print(f"  Multi-metric (PSNR+SSIM tau020): {gate['multi_metric']['psnr_ssim_tau020_advantage_ms']}ms")
    print(f"  Full metric (PSNR+SSIM+LPIPS tau020): {gate['multi_metric']['full_metric_tau020_advantage_ms']}ms")
    print(f"  Robust at max tau: {gate['ssim_only']['robust_at_max_tau']}")
    print(f"  Highest value missing: {gate['unknown_sensitivity']['highest_value_missing_measurement']} ({gate['unknown_sensitivity']['max_effect_on_advantage_ms']}ms)")
    print(f"  Cost model: {gate['cost_model']['realized_scene_cost_available']}")
    print(f"  Predictor search justified: {gate['predictor_search_justified']}")
    print(f"  GPU measurement justified: {gate['gpu_measurement_justified']}")
    if gate["if_gpu_measurement_justified"]:
        print(f"  Single highest value: {gate['if_gpu_measurement_justified']['single_highest_value_measurement']}")
    print(f"\n  Rationale: {gate['rationale'][:200]}...")

    return gate


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("C42++ B6 — Oracle Robustness Gate")
    print("=" * 70)

    b6a = phase_b6a()
    b6b = phase_b6b(b6a)
    b6c = phase_b6c()
    b6d = phase_b6d()
    b6e = phase_b6e()
    b6f = phase_b6f()
    gate = final_gate(b6a, b6b, b6c, b6d, b6e, b6f)

    print("\n" + "=" * 70)
    print("B6 ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
