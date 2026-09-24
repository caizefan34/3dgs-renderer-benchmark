#!/usr/bin/env python3
"""C42++ Adaptive Structural Supervision — CORRECTED B2-B5.

Corrected constrained-optimization framework:
- Adaptive oracle: per-scene cheapest scale satisfying |dSSIM| <= tau
- Fixed comparator: cheapest GLOBALLY FEASIBLE scale (intersection of all scenes' feasible sets)
- NEVER compare oracle against fixed-0.5 when fixed-0.5 violates the constraint
- UNKNOWN scales (missing measurements) are neither feasible nor infeasible
- Predictor metrics: constraint violation rate + feasible cost regret (NOT scale-label accuracy)
- Two problem formulations: Speed-first (Mode A) vs Quality-constrained (Mode B)
"""
import json, io, math
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
# DATA: 30K SSIM values (from canonical evidence)
# ============================================================

# Source: results/reference_v1/s23/pareto_summary.json + S2.2 report for Room
SSIM_30K = {
    "room": {
        "1.0":  0.9263,   # S2.2 report
        "0.75": None,     # UNKNOWN — never tested
        "0.625": None,    # UNKNOWN — never tested
        "0.5":  0.9185,   # S2.2 report
    },
    "garden": {
        "1.0":  0.8994,   # s22/baseline_30k
        "0.75": 0.8811,   # s23/s0.750_30k
        "0.625": None,    # UNKNOWN — 10K screening only
        "0.5":  0.8770,   # s22/c42_30k
    },
    "bicycle": {
        "1.0":  0.8383,   # s22/baseline_30k  (pareto_summary says 0.8383)
        "0.75": None,     # UNKNOWN — 10K screening only
        "0.625": 0.8084,  # s23/s0.625_30k
        "0.5":  0.8170,   # s22/c42_30k  (pareto_summary says 0.8170)
    },
}

# 10K SSIM values (from screening data + 30K run checkpoints at iter=10000)
SSIM_10K = {
    "room": {
        "1.0":  None,     # No 10K checkpoint available for Room (no training JSON)
        "0.75": None,     # UNKNOWN
        "0.625": None,    # UNKNOWN
        "0.5":  None,     # No 10K checkpoint available for Room
    },
    "garden": {
        "1.0":  None,     # Need to extract from 30K run checkpoint at 10000
        "0.75": None,     # Need to extract from screening
        "0.625": None,    # Need to extract from screening
        "0.5":  None,     # Need to extract from 30K run checkpoint at 10000
    },
    "bicycle": {
        "1.0":  None,     # Need to extract from 30K run checkpoint at 10000
        "0.75": None,     # Need to extract from screening
        "0.625": None,    # Need to extract from screening
        "0.5":  None,     # Need to extract from 30K run checkpoint at 10000
    },
}

# Try to extract 10K SSIM from the existing evidence
try:
    ev = load(RESULTS / "existing_evidence.json")
    for scene in ["garden", "bicycle"]:
        for scale_key in ev.get(scene, {}):
            data = ev[scene][scale_key]
            ckpts = data.get("checkpoints", [])
            for c in ckpts:
                if c["iteration"] == 10000:
                    # Map scale_key to canonical scale
                    if scale_key == "1.0":
                        SSIM_10K[scene]["1.0"] = c["ssim"]
                    elif scale_key == "0.5":
                        SSIM_10K[scene]["0.5"] = c["ssim"]
                    elif scale_key == "0.75":
                        SSIM_10K[scene]["0.75"] = c["ssim"]
                    elif scale_key == "0.625":
                        SSIM_10K[scene]["0.625"] = c["ssim"]
                    elif scale_key == "0.75_screening":
                        SSIM_10K[scene]["0.75"] = c["ssim"]
                    elif scale_key == "0.625_screening":
                        SSIM_10K[scene]["0.625"] = c["ssim"]
except Exception as e:
    print(f"Warning: could not extract 10K data from evidence: {e}")

# Print 10K data for verification
print("=== 10K SSIM values extracted ===")
for scene in SSIM_10K:
    print(f"  {scene}: {SSIM_10K[scene]}")

# Loss costs (historical, fixed-tensor, A100, 1080p)
LOSS_COST = {
    "1.0":   33.15,   # ms
    "0.75":  19.41,
    "0.625": 13.92,
    "0.5":    9.24,
}

SCENES = ["room", "garden", "bicycle"]
SCALES = ["1.0", "0.75", "0.625", "0.5"]
TAUS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]


def delta_ssim(scene, scale, stage="30K"):
    """Return |SSIM_1.0 - SSIM_s| for scene at stage. None if UNKNOWN."""
    ssim_table = SSIM_30K if stage == "30K" else SSIM_10K
    s1 = ssim_table[scene]["1.0"]
    ss = ssim_table[scene][scale]
    if s1 is None or ss is None:
        return None  # UNKNOWN
    return abs(s1 - ss)


def feasible_scales(scene, tau, stage="30K"):
    """Return feasible scale set for scene at tau. Excludes UNKNOWN."""
    feasible = []
    for s in SCALES:
        d = delta_ssim(scene, s, stage)
        if d is not None and d <= tau:
            feasible.append(s)
    return feasible


def unknown_scales(scene, stage="30K"):
    """Return scales with UNKNOWN SSIM at stage."""
    ssim_table = SSIM_30K if stage == "30K" else SSIM_10K
    return [s for s in SCALES if ssim_table[scene][s] is None and s != "1.0"]


def cheapest_feasible(scene, tau, stage="30K"):
    """Return (scale, cost) of cheapest feasible scale for scene."""
    fs = feasible_scales(scene, tau, stage)
    if not fs:
        return None, None
    # Sort by cost (cheapest first)
    best = min(fs, key=lambda s: LOSS_COST[s])
    return best, LOSS_COST[best]


def global_feasible_scales(tau, stage="30K"):
    """Return intersection of all scenes' feasible scales (PROVEN feasible only)."""
    sets = [set(feasible_scales(s, tau, stage)) for s in SCENES]
    if not all(sets):
        return set()
    return set.intersection(*sets)


def best_fixed_feasible(tau, stage="30K"):
    """Return (scale, cost) of cheapest globally feasible fixed scale."""
    gfs = global_feasible_scales(tau, stage)
    if not gfs:
        return None, None
    best = min(gfs, key=lambda s: LOSS_COST[s])
    return best, LOSS_COST[best]


# ============================================================
# Phase B2 (CORRECTED): Constrained Oracle
# ============================================================

def phase_b2_corrected():
    oracle = {
        "phase": "B2_corrected",
        "description": "Constrained oracle: per-scene cheapest feasible scale vs best globally-feasible fixed policy",
        "problem_definition": {
            "feasible_set": "F_j(tau) = { s : |SSIM_1.0,j - SSIM_s,j| <= tau }",
            "adaptive_oracle": "s_j* = argmin_{s in F_j(tau)} C(s)",
            "global_feasible": "F_global(tau) = intersection of all F_j(tau)",
            "fixed_comparator": "s_fixed* = argmin_{s in F_global(tau)} C(s)",
            "key_correction": "NEVER compare oracle against fixed-0.5 when fixed-0.5 violates the constraint. Only compare against best FIXED FEASIBLE policy.",
            "unknown_handling": "Scales with missing SSIM measurements are marked UNKNOWN and excluded from feasible sets. They are neither feasible nor infeasible.",
        },
        "ssim_30k_source": "results/reference_v1/s23/pareto_summary.json + S2.2 report for Room",
        "loss_cost_source": "results/reference_v1/s23/fixed_tensor_loss_benchmark.json (historical, fixed-tensor, A100)",
        "delta_ssim_30k": {},
        "tau_sweep_30k": {},
        "stage_dependence": {},
    }

    # Record delta SSIM values
    for scene in SCENES:
        oracle["delta_ssim_30k"][scene] = {}
        for s in SCALES:
            if s == "1.0":
                oracle["delta_ssim_30k"][scene][s] = 0.0
            else:
                d = delta_ssim(scene, s, "30K")
                if d is None:
                    oracle["delta_ssim_30k"][scene][s] = "UNKNOWN"
                else:
                    oracle["delta_ssim_30k"][scene][s] = round(d, 4)

    # Tau sweep at 30K
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        entry = {
            "tau": tau,
            "per_scene": {},
            "adaptive_oracle": {},
            "global_feasible_scales": [],
            "best_fixed_feasible": {},
        }

        oracle_total_cost = 0
        n_scenes = 0

        for scene in SCENES:
            fs = feasible_scales(scene, tau, "30K")
            unk = unknown_scales(scene, "30K")
            best_s, best_c = cheapest_feasible(scene, tau, "30K")

            entry["per_scene"][scene] = {
                "feasible_scales": fs,
                "unknown_scales": unk,
                "infeasible_scales": [s for s in SCALES if s not in fs and s not in unk and s != "1.0"],
                "oracle_scale": best_s,
                "oracle_cost_ms": best_c,
                "delta_ssim_at_oracle": round(delta_ssim(scene, best_s, "30K"), 4) if best_s else None,
            }

            if best_c is not None:
                oracle_total_cost += best_c
                n_scenes += 1

        # Adaptive oracle average cost
        avg_oracle_cost = oracle_total_cost / n_scenes if n_scenes > 0 else None
        entry["adaptive_oracle"] = {
            "per_scene_scales": {s: entry["per_scene"][s]["oracle_scale"] for s in SCENES},
            "per_scene_costs": {s: entry["per_scene"][s]["oracle_cost_ms"] for s in SCENES},
            "average_cost_ms": round(avg_oracle_cost, 2) if avg_oracle_cost else None,
            "total_cost_ms": round(oracle_total_cost, 2),
        }

        # Global feasible fixed policy
        gfs = sorted(global_feasible_scales(tau, "30K"))
        entry["global_feasible_scales"] = gfs

        bf_scale, bf_cost = best_fixed_feasible(tau, "30K")
        entry["best_fixed_feasible"] = {
            "scale": bf_scale,
            "cost_ms": bf_cost,
        }

        # Oracle saving vs best fixed feasible
        if avg_oracle_cost is not None and bf_cost is not None:
            saving = bf_cost - avg_oracle_cost
            pct = (saving / bf_cost * 100) if bf_cost > 0 else 0
            entry["oracle_saving"] = {
                "absolute_ms": round(saving, 2),
                "percentage": round(pct, 1),
                "interpretation": "POSITIVE — adaptive oracle is cheaper" if saving > 0.01 else
                                  ("ZERO — no advantage" if abs(saving) < 0.01 else
                                   "NEGATIVE — fixed is cheaper (should not happen)"),
            }
        else:
            entry["oracle_saving"] = {"absolute_ms": None, "percentage": None, "interpretation": "INCONCLUSIVE — missing data"}

        oracle["tau_sweep_30k"][tau_str] = entry

    # Stage dependence analysis (10K vs 30K)
    stage_dep = {"10K": {}, "30K": {}}
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        for stage in ["10K", "30K"]:
            entry = {}
            for scene in SCENES:
                fs = feasible_scales(scene, tau, stage)
                best_s, best_c = cheapest_feasible(scene, tau, stage)
                entry[scene] = {
                    "feasible": fs,
                    "oracle_scale": best_s,
                    "oracle_cost": best_c,
                }
            stage_dep[stage][tau_str] = entry

    # Check if oracle changes between 10K and 30K
    changes = []
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        for scene in SCENES:
            s10 = stage_dep["10K"][tau_str][scene]["oracle_scale"]
            s30 = stage_dep["30K"][tau_str][scene]["oracle_scale"]
            if s10 is not None and s30 is not None and s10 != s30:
                changes.append({
                    "scene": scene,
                    "tau": tau,
                    "10K_scale": s10,
                    "30K_scale": s30,
                    "direction": "scale increases (more conservative) at 30K" if float(s30) > float(s10) else "scale decreases at 30K",
                })

    oracle["stage_dependence"] = {
        "10K_oracles": stage_dep["10K"],
        "30K_oracles": stage_dep["30K"],
        "changes": changes,
        "summary": f"{len(changes)} stage-dependent oracle changes detected" if changes else "No stage-dependent changes detected",
    }

    # Max oracle opportunity
    max_saving = 0
    max_tau = None
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        s = oracle["tau_sweep_30k"][tau_str].get("oracle_saving", {})
        abs_saving = s.get("absolute_ms", 0)
        if abs_saving and abs_saving > max_saving:
            max_saving = abs_saving
            max_tau = tau

    oracle["max_oracle_opportunity"] = {
        "tau": max_tau,
        "saving_ms": max_saving,
        "saving_pct": oracle["tau_sweep_30k"][f"{max_tau:.3f}"]["oracle_saving"]["percentage"] if max_tau else None,
    }

    save(RESULTS / "oracle_schedule.json", oracle)
    print("\n=== B2 CORRECTED ORACLE ===")
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        e = oracle["tau_sweep_30k"][tau_str]
        ao = e["adaptive_oracle"]
        bf = e["best_fixed_feasible"]
        sv = e["oracle_saving"]
        per_scene = {s: e["per_scene"][s]["oracle_scale"] for s in SCENES}
        print(f"  tau={tau:.3f}: oracle={per_scene}, avg_cost={ao['average_cost_ms']}, "
              f"fixed_feasible={bf['scale']}({bf['cost_ms']}), saving={sv['absolute_ms']}ms ({sv['percentage']}%)")
    print(f"\n  Max opportunity: tau={oracle['max_oracle_opportunity']['tau']}, "
          f"saving={oracle['max_oracle_opportunity']['saving_ms']}ms "
          f"({oracle['max_oracle_opportunity']['saving_pct']}%)")
    print(f"\n  Stage dependence: {len(changes)} changes")
    for c in changes:
        print(f"    {c['scene']} tau={c['tau']}: {c['10K_scale']} -> {c['30K_scale']} ({c['direction']})")

    return oracle


# ============================================================
# Phase B3 (CORRECTED): Predictor Evaluation
# ============================================================

def phase_b3_corrected(oracle):
    """Evaluate predictors using constraint-based metrics, NOT scale-label accuracy."""

    # HF predictor: predicts based on HF fraction threshold
    # Garden HF=0.766 -> predict 0.75, Bicycle HF=0.561 -> predict 0.625, Room (no data) -> default 0.5
    HF_PREDICTIONS = {
        "room": "0.5",      # default (no HF data)
        "garden": "0.75",   # HF=0.766 > 0.65
        "bicycle": "0.625", # HF=0.561, 0.45 < HF < 0.65
    }

    # Fixed-0.5 policy
    FIXED_05 = {s: "0.5" for s in SCENES}

    # Fixed-0.75 policy
    FIXED_075 = {s: "0.75" for s in SCENES}

    predictor = {
        "phase": "B3_corrected",
        "description": "Predictor evaluation via constraint violation rate + feasible cost regret",
        "metric_definitions": {
            "constraint_violation_rate": "fraction of scenes where predicted scale violates tau (|dSSIM| > tau or UNKNOWN)",
            "feasible_cost_regret": "For predictions satisfying constraint: C(pred) - C(oracle). Averages only over non-violating scenes.",
            "expected_constrained_cost": "Average total cost = sum of C(pred) for non-violating + penalty for violating. Here we report avg C(pred) over ALL scenes (violating scenes still incur their compute cost).",
            "deleted_metric": "Oracle scale-label accuracy is NO LONGER USED. It was invalid because different scales can have identical cost-rank under constraint.",
        },
        "policies": {},
    }

    policies = {
        "HF_predictor": HF_PREDICTIONS,
        "fixed_0.5": FIXED_05,
        "fixed_0.75": FIXED_075,
    }

    for policy_name, predictions in policies.items():
        policy_eval = {"predictions": predictions, "tau_results": {}}

        for tau in TAUS:
            tau_str = f"{tau:.3f}"
            violations = []
            regrets = []
            costs = []

            for scene in SCENES:
                pred_scale = predictions[scene]
                d = delta_ssim(scene, pred_scale, "30K")
                pred_cost = LOSS_COST[pred_scale]

                # Get oracle for this scene/tau
                oracle_entry = oracle["tau_sweep_30k"][tau_str]["per_scene"][scene]
                oracle_scale = oracle_entry["oracle_scale"]
                oracle_cost = oracle_entry["oracle_cost_ms"]

                if d is None:
                    # UNKNOWN — cannot prove constraint satisfaction
                    violations.append({
                        "scene": scene,
                        "predicted_scale": pred_scale,
                        "reason": "UNKNOWN delta SSIM — constraint satisfaction cannot be verified",
                    })
                    costs.append(pred_cost)
                elif d > tau:
                    violations.append({
                        "scene": scene,
                        "predicted_scale": pred_scale,
                        "delta_ssim": round(d, 4),
                        "tau": tau,
                        "excess": round(d - tau, 4),
                    })
                    costs.append(pred_cost)
                else:
                    # Constraint satisfied — compute regret
                    regret = pred_cost - oracle_cost if oracle_cost else None
                    regrets.append({
                        "scene": scene,
                        "predicted_scale": pred_scale,
                        "oracle_scale": oracle_scale,
                        "regret_ms": round(regret, 2) if regret is not None else None,
                    })
                    costs.append(pred_cost)

            violation_rate = len(violations) / len(SCENES)
            avg_regret = sum(r["regret_ms"] for r in regrets if r["regret_ms"] is not None) / max(1, len(regrets)) if regrets else None
            expected_cost = sum(costs) / len(costs)

            # Get oracle avg cost and best fixed feasible for comparison
            oracle_avg = oracle["tau_sweep_30k"][tau_str]["adaptive_oracle"]["average_cost_ms"]
            bf_scale, bf_cost = best_fixed_feasible(tau, "30K")

            policy_eval["tau_results"][tau_str] = {
                "constraint_violation_rate": round(violation_rate, 4),
                "n_violations": len(violations),
                "violations": violations,
                "feasible_cost_regret_avg_ms": round(avg_regret, 2) if avg_regret is not None else None,
                "feasible_regrets": regrets,
                "expected_cost_ms": round(expected_cost, 2),
                "oracle_avg_cost_ms": oracle_avg,
                "best_fixed_feasible_cost_ms": bf_cost,
                "cost_vs_oracle_ms": round(expected_cost - oracle_avg, 2) if oracle_avg else None,
                "cost_vs_fixed_feasible_ms": round(expected_cost - bf_cost, 2) if bf_cost else None,
            }

        predictor["policies"][policy_name] = policy_eval

    # Summary: which policy is best at each tau?
    predictor["summary"] = {}
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        best_policy = None
        best_cost = float("inf")
        for pname, peval in predictor["policies"].items():
            tr = peval["tau_results"][tau_str]
            # Primary criterion: lowest violation rate, then lowest expected cost
            vr = tr["constraint_violation_rate"]
            ec = tr["expected_cost_ms"]
            if vr == 0 and ec < best_cost:
                best_cost = ec
                best_policy = pname
            elif best_policy is None and vr > 0:
                # All policies violate — pick lowest cost among zero-violation if any, else lowest violation
                pass

        # Recompute: find policy with 0 violations and lowest cost; if none, note
        zero_violation = [(pname, peval["tau_results"][tau_str]["expected_cost_ms"])
                         for pname, peval in predictor["policies"].items()
                         if peval["tau_results"][tau_str]["constraint_violation_rate"] == 0]

        if zero_violation:
            best_zero = min(zero_violation, key=lambda x: x[1])
            predictor["summary"][tau_str] = {
                "best_zero_violation_policy": best_zero[0],
                "best_zero_violation_cost": best_zero[1],
            }
        else:
            # All policies violate — report lowest violation rate
            min_vr = min(peval["tau_results"][tau_str]["constraint_violation_rate"]
                        for peval in predictor["policies"].values())
            best_vr_policies = [pname for pname, peval in predictor["policies"].items()
                               if peval["tau_results"][tau_str]["constraint_violation_rate"] == min_vr]
            predictor["summary"][tau_str] = {
                "best_zero_violation_policy": None,
                "note": f"All policies violate at tau={tau}. Lowest violation rate={min_vr} by {best_vr_policies}",
            }

    save(RESULTS / "predictor_results.json", predictor)
    print("\n=== B3 CORRECTED PREDICTOR ===")
    for pname, peval in predictor["policies"].items():
        print(f"\n  {pname}:")
        for tau in TAUS:
            tau_str = f"{tau:.3f}"
            tr = peval["tau_results"][tau_str]
            print(f"    tau={tau:.3f}: violations={tr['n_violations']}/{len(SCENES)}, "
                  f"expected_cost={tr['expected_cost_ms']}ms, "
                  f"regret={tr['feasible_cost_regret_avg_ms']}ms, "
                  f"vs_fixed_feasible={tr['cost_vs_fixed_feasible_ms']}ms")

    return predictor


# ============================================================
# Phase B4 (CORRECTED): Leave-One-Scene-Out
# ============================================================

def phase_b4_corrected(oracle, predictor):
    """LOSO with constraint-based metrics."""

    loso = {
        "phase": "B4_corrected",
        "description": "LOSO validation using constraint violation rate + feasible cost regret",
        "folds": {},
        "summary": {},
    }

    # HF data available: Garden (0.766), Bicycle (0.561). Room: no HF data.
    HF_VALUES = {
        "garden": 0.766,
        "bicycle": 0.561,
    }

    for held_out in SCENES:
        train_scenes = [s for s in SCENES if s != held_out]

        fold = {"held_out": held_out, "train_scenes": train_scenes}

        # Train HF threshold on 2 scenes, predict on held-out
        train_hfs = {s: HF_VALUES[s] for s in train_scenes if s in HF_VALUES}

        if len(train_hfs) >= 2:
            hf_vals = sorted(train_hfs.values())
            threshold = (hf_vals[0] + hf_vals[1]) / 2

            held_hf = HF_VALUES.get(held_out)
            if held_hf is not None:
                pred = "0.75" if held_hf > threshold else "0.5"
            else:
                pred = "0.5"  # safe default when no HF data (Room)

            fold["hf_rule"] = {
                "train_hfs": train_hfs,
                "threshold": threshold,
                "held_out_hf": held_hf,
                "prediction": pred,
            }
        elif held_out in HF_VALUES:
            # Only 1 train scene has HF — can't set threshold
            fold["hf_rule"] = {
                "note": "Insufficient HF data for threshold training",
                "prediction": "0.5",  # conservative default
            }
        else:
            fold["hf_rule"] = {
                "note": "No HF data for held-out scene, defaulting to 0.5",
                "prediction": "0.5",
            }

        # Evaluate this fold's prediction at each tau
        pred_scale = fold["hf_rule"]["prediction"]
        fold["tau_evaluation"] = {}

        for tau in TAUS:
            tau_str = f"{tau:.3f}"
            d = delta_ssim(held_out, pred_scale, "30K")
            oracle_entry = oracle["tau_sweep_30k"][tau_str]["per_scene"][held_out]
            oracle_scale = oracle_entry["oracle_scale"]
            oracle_cost = oracle_entry["oracle_cost_ms"]

            pred_cost = LOSS_COST[pred_scale]

            if d is None:
                violation = True
                reason = "UNKNOWN"
                regret = None
            elif d > tau:
                violation = True
                reason = f"delta={d:.4f} > tau={tau}"
                regret = None
            else:
                violation = False
                reason = "satisfied"
                regret = pred_cost - oracle_cost if oracle_cost else None

            # Also evaluate fixed-0.5 for this fold
            d05 = delta_ssim(held_out, "0.5", "30K")
            if d05 is None:
                f05_violation = True
                f05_reason = "UNKNOWN"
            elif d05 > tau:
                f05_violation = True
                f05_reason = f"delta={d05:.4f} > tau={tau}"
            else:
                f05_violation = False
                f05_reason = "satisfied"

            fold["tau_evaluation"][tau_str] = {
                "hf_prediction": pred_scale,
                "hf_violation": violation,
                "hf_reason": reason,
                "hf_regret_ms": round(regret, 2) if regret is not None else None,
                "hf_cost_ms": pred_cost,
                "oracle_scale": oracle_scale,
                "oracle_cost_ms": oracle_cost,
                "fixed_05_violation": f05_violation,
                "fixed_05_reason": f05_reason,
            }

        loso["folds"][held_out] = fold

    # Summary across folds
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        hf_violations = sum(1 for s in SCENES if loso["folds"][s]["tau_evaluation"][tau_str]["hf_violation"])
        f05_violations = sum(1 for s in SCENES if loso["folds"][s]["tau_evaluation"][tau_str]["fixed_05_violation"])
        loso["summary"][tau_str] = {
            "hf_violation_rate": round(hf_violations / len(SCENES), 4),
            "fixed_05_violation_rate": round(f05_violations / len(SCENES), 4),
            "note": "HF rule and fixed-0.5 are evaluated by constraint satisfaction, NOT scale-label accuracy",
        }

    loso["key_limitation"] = (
        "Only 2 scenes (Garden, Bicycle) have HF fraction data. LOSO with 2 training scenes is weak. "
        "Room has no HF probe — any HF-based rule defaults to 0.5 for Room, which is correct at tau >= 0.010 "
        "but may not generalize to new scenes without HF data."
    )

    save(RESULTS / "leave_one_scene_out.json", loso)
    print("\n=== B4 CORRECTED LOSO ===")
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        s = loso["summary"][tau_str]
        print(f"  tau={tau:.3f}: HF_violation_rate={s['hf_violation_rate']}, fixed_0.5_violation_rate={s['fixed_05_violation_rate']}")

    return loso


# ============================================================
# Phase B5 (CORRECTED): Opportunity Estimate
# ============================================================

def phase_b5_corrected(oracle, predictor, loso):
    """Opportunity estimate: oracle vs best FIXED FEASIBLE (not vs infeasible fixed-0.5)."""

    opp = {
        "phase": "B5_corrected",
        "description": "Opportunity estimate with CORRECT fixed comparator (best globally-feasible fixed policy)",
        "loss_costs_ms": LOSS_COST,
        "loss_cost_source": "results/reference_v1/s23/fixed_tensor_loss_benchmark.json (historical, fixed-tensor, A100, 1080p)",
        "tau_sweep": {},
        "two_mode_analysis": {},
    }

    # Tau sweep
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        e = oracle["tau_sweep_30k"][tau_str]
        ao = e["adaptive_oracle"]
        bf = e["best_fixed_feasible"]
        sv = e["oracle_saving"]

        opp["tau_sweep"][tau_str] = {
            "tau": tau,
            "per_scene_oracle": {s: e["per_scene"][s]["oracle_scale"] for s in SCENES},
            "per_scene_oracle_cost": {s: e["per_scene"][s]["oracle_cost_ms"] for s in SCENES},
            "adaptive_oracle_avg_cost_ms": ao["average_cost_ms"],
            "global_feasible_scales": e["global_feasible_scales"],
            "best_fixed_feasible_scale": bf["scale"],
            "best_fixed_feasible_cost_ms": bf["cost_ms"],
            "oracle_saving_ms": sv["absolute_ms"],
            "oracle_saving_pct": sv["percentage"],
            "interpretation": sv["interpretation"],
        }

    # Two-mode analysis
    # Mode A: Speed-first (accept measured quality tradeoff)
    # The best speed-first policy is fixed-0.5 (fastest Pareto-optimal on all scenes)
    # Report its quality tradeoff
    speed_first = {
        "mode": "A_speed_first",
        "objective": "maximize training throughput, accept measured quality tradeoff",
        "best_policy": "fixed_0.5",
        "loss_cost_ms": LOSS_COST["0.5"],
        "throughput_range": "1.60x-1.90x across all scenes",
        "quality_tradeoff": {
            "room": {"delta_psnr": 0.24, "delta_ssim": -0.0078, "note": "Scale 0.5 is STRICTLY BETTER (higher PSNR, fewer Gaussians)"},
            "garden": {"delta_psnr": -0.40, "delta_ssim": -0.0224, "note": "Quality degradation accepted for 1.68x throughput"},
            "bicycle": {"delta_psnr": -0.04, "delta_ssim": -0.0213, "note": "Negligible PSNR loss, SSIM degradation, 1.60x throughput"},
        },
        "interpretation": "Fixed-0.5 is the fastest tested common Pareto operating point with scene-dependent quality degradation. It is NOT a quality-constrained policy when tau < 0.025.",
    }

    # Mode B: Quality-constrained
    # The adaptive oracle's value depends on tau
    quality_constrained = {
        "mode": "B_quality_constrained",
        "objective": "minimize supervision cost subject to |dSSIM| <= tau",
        "oracle_opportunity": "PASS" if oracle["max_oracle_opportunity"]["saving_ms"] > 0.5 else "FAIL",
        "max_oracle_saving": {
            "tau": oracle["max_oracle_opportunity"]["tau"],
            "saving_ms": oracle["max_oracle_opportunity"]["saving_ms"],
            "saving_pct": oracle["max_oracle_opportunity"]["saving_pct"],
        },
        "tau_range_with_opportunity": [],
        "tau_range_without_opportunity": [],
    }

    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        sv = oracle["tau_sweep_30k"][tau_str]["oracle_saving"]
        saving = sv["absolute_ms"]
        if saving and saving > 0.5:
            quality_constrained["tau_range_with_opportunity"].append({
                "tau": tau,
                "saving_ms": saving,
                "saving_pct": sv["percentage"],
            })
        else:
            quality_constrained["tau_range_without_opportunity"].append({
                "tau": tau,
                "saving_ms": saving,
                "reason": "Oracle and best fixed feasible coincide" if saving and abs(saving) < 0.5 else "No feasible non-1.0 scale",
            })

    opp["two_mode_analysis"] = {
        "mode_A_speed_first": speed_first,
        "mode_B_quality_constrained": quality_constrained,
        "key_separation": "Mode A (speed-first) and Mode B (quality-constrained) are DIFFERENT objectives. Fixed-0.5 is optimal for Mode A but may violate Mode B constraints. The adaptive oracle is only relevant for Mode B.",
    }

    # Current predictor status
    # At the max-opportunity tau, check HF predictor violation rate
    max_tau = oracle["max_oracle_opportunity"]["tau"]
    if max_tau:
        max_tau_str = f"{max_tau:.3f}"
        hf_at_max = predictor["policies"]["HF_predictor"]["tau_results"][max_tau_str]
        opp["predictor_at_max_opportunity_tau"] = {
            "tau": max_tau,
            "hf_violation_rate": hf_at_max["constraint_violation_rate"],
            "hf_expected_cost_ms": hf_at_max["expected_cost_ms"],
            "oracle_avg_cost_ms": hf_at_max["oracle_avg_cost_ms"],
            "best_fixed_feasible_cost_ms": hf_at_max["best_fixed_feasible_cost_ms"],
            "verdict": "FAIL" if hf_at_max["constraint_violation_rate"] > 0 else "PASS",
        }

    # Counterfactual limitation
    opp["counterfactual_limitation"] = (
        "The offline oracle is constructed from independently trained fixed-scale trajectories. "
        "It is not yet evidence that switching scales online would follow the same trajectory. "
        "Therefore no full 30K adaptive training should be launched yet."
    )

    save(RESULTS / "opportunity_estimate.json", opp)
    print("\n=== B5 CORRECTED OPPORTUNITY ===")
    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        e = opp["tau_sweep"][tau_str]
        print(f"  tau={tau:.3f}: oracle_avg={e['adaptive_oracle_avg_cost_ms']}ms, "
              f"fixed_feasible={e['best_fixed_feasible_scale']}({e['best_fixed_feasible_cost_ms']}ms), "
              f"saving={e['oracle_saving_ms']}ms ({e['oracle_saving_pct']}%)")
    print(f"\n  Max opportunity: tau={opp['two_mode_analysis']['mode_B_quality_constrained']['max_oracle_saving']['tau']}, "
          f"saving={opp['two_mode_analysis']['mode_B_quality_constrained']['max_oracle_saving']['saving_ms']}ms "
          f"({opp['two_mode_analysis']['mode_B_quality_constrained']['max_oracle_saving']['saving_pct']}%)")
    print(f"  Oracle opportunity: {opp['two_mode_analysis']['mode_B_quality_constrained']['oracle_opportunity']}")
    if "predictor_at_max_opportunity_tau" in opp:
        p = opp["predictor_at_max_opportunity_tau"]
        print(f"  HF predictor at max-opportunity tau: violation_rate={p['hf_violation_rate']}, verdict={p['verdict']}")

    return opp


# ============================================================
# Final Decision (CORRECTED)
# ============================================================

def final_decision_corrected(oracle, predictor, loso, opp):
    decision = {
        "experiment": "C42_ADAPTIVE_REANALYSIS",
        "title": "C42++ Adaptive Structural Supervision — Corrected Constrained Analysis",
        "date": "2026-09-15",
        "correction": "Previous DROP decision was based on invalid comparison: quality-constrained oracle vs infeasible fixed-0.5. Corrected to compare oracle vs best globally-feasible fixed policy.",
        "problem_formulation": {
            "mode_A_speed_first": "Maximize throughput, accept quality tradeoff. Best policy: fixed-0.5.",
            "mode_B_quality_constrained": "Minimize cost subject to |dSSIM| <= tau. This is where adaptive resolution may have value.",
        },
        "oracle_summary": {},
        "predictor_summary": {},
        "classification": {},
        "decision": "",
        "rationale": "",
        "counterfactual_limitation": opp["counterfactual_limitation"],
    }

    # Oracle summary
    max_opp = oracle["max_oracle_opportunity"]
    decision["oracle_summary"] = {
        "max_oracle_saving_ms": max_opp["saving_ms"],
        "max_oracle_saving_pct": max_opp["saving_pct"],
        "max_opportunity_tau": max_opp["tau"],
        "opportunity_pass_threshold_ms": 0.5,
        "verdict": "PASS — oracle materially better than best fixed feasible" if max_opp["saving_ms"] > 0.5 else "FAIL",
    }

    # Predictor summary at max-opportunity tau
    if "predictor_at_max_opportunity_tau" in opp:
        p = opp["predictor_at_max_opportunity_tau"]
        decision["predictor_summary"] = {
            "at_max_opportunity_tau": p["tau"],
            "constraint_violation_rate": p["hf_violation_rate"],
            "verdict": p["verdict"],
        }
    else:
        decision["predictor_summary"] = {"verdict": "INCONCLUSIVE"}

    # Classification
    oracle_pass = max_opp["saving_ms"] > 0.5
    predictor_pass = decision["predictor_summary"].get("verdict") == "PASS"

    if not oracle_pass:
        decision["classification"]["ORACLE_OPPORTUNITY"] = "FAIL"
        decision["classification"]["CURRENT_PREDICTOR"] = "FAIL" if not predictor_pass else "PASS"
        decision["classification"]["ADAPTIVE_METHOD"] = "DROP"
        decision["decision"] = "DROP"
        decision["rationale"] = "Oracle provides no material advantage over best fixed feasible policy."
    elif not predictor_pass:
        decision["classification"]["ORACLE_OPPORTUNITY"] = "PASS"
        decision["classification"]["CURRENT_PREDICTOR"] = "FAIL"
        decision["classification"]["ADAPTIVE_METHOD"] = "MODIFY"
        decision["decision"] = "MODIFY"
        decision["rationale"] = (
            f"Oracle opportunity is REAL (max saving {max_opp['saving_ms']}ms = {max_opp['saving_pct']}% at tau={max_opp['tau']}), "
            f"but the current HF predictor FAILS (constraint violation rate={p['hf_violation_rate']} at max-opportunity tau). "
            "A better cheap signal is needed. The oracle opportunity justifies continued investigation."
        )
    else:
        decision["classification"]["ORACLE_OPPORTUNITY"] = "PASS"
        decision["classification"]["CURRENT_PREDICTOR"] = "PASS"
        decision["classification"]["ADAPTIVE_METHOD"] = "KEEP"
        decision["decision"] = "KEEP"
        decision["rationale"] = "Oracle opportunity is real and predictor works."

    # Mechanism wording correction
    decision["mechanism_wording_correction"] = (
        "The anomalous multi-resolution gradient alignment (Bicycle scale=0.75 cosine=0.506 with rel_l2=8.19) "
        "is consistent with non-monotonic scale-dependent optimization dynamics, but causality has not been established. "
        "The previous report's claim that this anomaly 'explains' the 0.625-vs-0.5 trajectory was too strong."
    )

    # Unknown scale impact
    decision["unknown_scale_impact"] = {
        "room_0.75_0.625": "UNKNOWN — never tested. If feasible at some tau, Room oracle could be cheaper than 0.5. Current oracle is an UPPER BOUND on achievable cost.",
        "garden_0.625": "UNKNOWN at 30K — 10K screening only. If feasible, Garden oracle could pick 0.625 (13.92ms) instead of 0.75 (19.41ms).",
        "bicycle_0.75": "UNKNOWN at 30K — 10K screening only. If feasible, Bicycle could have more options.",
        "impact_on_oracle": "The computed oracle saving is a LOWER BOUND — true opportunity could be larger if UNKNOWN scales are feasible.",
        "impact_on_fixed": "The computed best-fixed-feasible may be conservative — more scales might be globally feasible if UNKNOWN measurements were available.",
    }

    save(RESULTS / "final_decision.json", decision)
    print(f"\n=== FINAL DECISION: {decision['decision']} ===")
    print(f"Oracle: {decision['classification']['ORACLE_OPPORTUNITY']}")
    print(f"Predictor: {decision['classification']['CURRENT_PREDICTOR']}")
    print(f"Method: {decision['classification']['ADAPTIVE_METHOD']}")
    print(f"Rationale: {decision['rationale']}")

    return decision


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("C42++ CORRECTED Analysis (B2-B5)")
    print("Constrained optimization: oracle vs best FIXED FEASIBLE")
    print("=" * 70)

    oracle = phase_b2_corrected()
    predictor = phase_b3_corrected(oracle)
    loso = phase_b4_corrected(oracle, predictor)
    opp = phase_b5_corrected(oracle, predictor, loso)
    decision = final_decision_corrected(oracle, predictor, loso, opp)

    print("\n" + "=" * 70)
    print("CORRECTED ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
