#!/usr/bin/env python3
"""C42++ B7 — Falsification-First Missing-Scale Gate.

B7-A: Three VOI notions (nominal, max conditional swing, falsification value)
B7-B: Explicit global-0.75 feasibility audit
B7-C: Garden 0.625 arithmetic check
B7-D: Rank training experiments by multiple criteria
B7-E: Checkpoint/resume availability

Pure offline CPU analysis. No GPU, no training.
"""
import json, io, itertools
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results" / "c42_adaptive"
RESULTS.mkdir(parents=True, exist_ok=True)

def save(path, data):
    with io.open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ============================================================
# DATA (tau=0.020 focus)
# ============================================================

SSIM_30K = {
    "room":    {"1.0": 0.9263, "0.75": None,  "0.625": None,  "0.5": 0.9185},
    "garden":  {"1.0": 0.8994, "0.75": 0.8811, "0.625": None,  "0.5": 0.8770},
    "bicycle": {"1.0": 0.8383, "0.75": None,  "0.625": 0.8084, "0.5": 0.8170},
}

LOSS_COST = {"1.0": 33.15, "0.75": 19.41, "0.625": 13.92, "0.5": 9.24}

SCENES = ["room", "garden", "bicycle"]
SCALES = ["1.0", "0.75", "0.625", "0.5"]
TAU = 0.020

UNKNOWN_VARS = ["room_0.75", "room_0.625", "garden_0.625", "bicycle_0.75"]
# Map to (scene, scale)
VAR_MAP = {
    "room_0.75": ("room", "0.75"),
    "room_0.625": ("room", "0.625"),
    "garden_0.625": ("garden", "0.625"),
    "bicycle_0.75": ("bicycle", "0.75"),
}


def abs_delta_ssim(scene, scale):
    """|SSIM_1.0 - SSIM_s|. None if UNKNOWN."""
    s1 = SSIM_30K[scene]["1.0"]
    ss = SSIM_30K[scene][scale]
    if s1 is None or ss is None:
        return None
    return abs(s1 - ss)


def compute_advantage(assignment):
    """Compute adaptive advantage for a given UNKNOWN assignment at tau=0.020.

    assignment: dict of var_name -> True (feasible) or False (infeasible)
    Returns: (oracle_scales, oracle_avg_cost, global_feasible, best_fixed_cost, advantage)
    """
    # Build per-scene feasible sets
    scene_feasible = {}
    for scene in SCENES:
        feasible = []
        for s in SCALES:
            if s == "1.0":
                feasible.append(s)
                continue
            d = abs_delta_ssim(scene, s)
            if d is None:
                # Check if this is an UNKNOWN variable in our assignment
                key = f"{scene}_{s}"
                if key in assignment:
                    if assignment[key]:
                        feasible.append(s)
                # If not in assignment, it's a KNOWN infeasible (e.g., bicycle_0.625 at tau=0.020)
            else:
                if d <= TAU:
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

    oracle_avg = sum(c for c in oracle_costs.values() if c is not None) / len(SCENES)

    # Global feasible: intersection
    global_feasible = set(scene_feasible["room"]) & set(scene_feasible["garden"]) & set(scene_feasible["bicycle"])

    if global_feasible:
        bf_scale = min(global_feasible, key=lambda s: LOSS_COST[s])
        bf_cost = LOSS_COST[bf_scale]
    else:
        bf_scale = None
        bf_cost = None

    advantage = bf_cost - oracle_avg if bf_cost and oracle_avg else None

    return {
        "oracle_scales": oracle_scales,
        "oracle_costs": oracle_costs,
        "oracle_avg_cost": round(oracle_avg, 2),
        "scene_feasible": {s: scene_feasible[s] for s in SCENES},
        "global_feasible": sorted(global_feasible),
        "best_fixed_scale": bf_scale,
        "best_fixed_cost": bf_cost,
        "advantage": round(advantage, 2) if advantage is not None else None,
    }


# ============================================================
# B7-A: Three VOI notions
# ============================================================

def phase_b7a():
    result = {
        "phase": "B7-A",
        "description": "Three VOI notions at tau=0.020: nominal, max conditional swing, falsification",
        "tau": TAU,
        "variables": {},
    }

    # Generate all 16 assignments
    all_assignments = []
    for bits in itertools.product([True, False], repeat=4):
        a = {UNKNOWN_VARS[i]: bits[i] for i in range(4)}
        r = compute_advantage(a)
        r["assignment"] = {k: "feasible" if v else "infeasible" for k, v in a.items()}
        all_assignments.append(r)

    # For each variable, compute three VOI notions
    for var in UNKNOWN_VARS:
        voi = {"nominal": None, "max_conditional_swing": None, "falsification_value": None}

        # 1. Nominal marginal effect: all others infeasible
        other_vars = [v for v in UNKNOWN_VARS if v != var]
        a_feasible = {v: False for v in other_vars}
        a_feasible[var] = True
        a_infeasible = {v: False for v in other_vars}
        a_infeasible[var] = False

        adv_feasible = compute_advantage(a_feasible)["advantage"]
        adv_infeasible = compute_advantage(a_infeasible)["advantage"]
        voi["nominal"] = round(adv_feasible - adv_infeasible, 2) if adv_feasible is not None and adv_infeasible is not None else None

        # 2. Max conditional swing: for each fixed assignment of other vars,
        #    compute |advantage(x=feasible, other) - advantage(x=infeasible, other)|
        max_swing = 0
        max_swing_detail = None
        for other_bits in itertools.product([True, False], repeat=3):
            other_a = {other_vars[i]: other_bits[i] for i in range(3)}

            a_f = dict(other_a)
            a_f[var] = True
            a_i = dict(other_a)
            a_i[var] = False

            adv_f = compute_advantage(a_f)["advantage"]
            adv_i = compute_advantage(a_i)["advantage"]

            if adv_f is not None and adv_i is not None:
                swing = abs(adv_f - adv_i)
                if swing > max_swing:
                    max_swing = swing
                    max_swing_detail = {
                        "other_assignment": {k: "feasible" if v else "infeasible" for k, v in other_a.items()},
                        "advantage_x_feasible": adv_f,
                        "advantage_x_infeasible": adv_i,
                        "swing": round(swing, 2),
                        "direction": "DECREASES advantage (falsification)" if adv_f < adv_i else "INCREASES advantage (anti-falsification)",
                    }

        voi["max_conditional_swing"] = round(max_swing, 2)
        voi["max_swing_detail"] = max_swing_detail

        # 3. Falsification value: can measuring x make a cheaper GLOBAL FIXED policy feasible?
        # Check if x can contribute to making 0.75 or 0.625 globally feasible
        scene, scale = VAR_MAP[var]
        falsification = "NONE"

        if scale == "0.75":
            # 0.75 global feasibility requires Room 0.75, Garden 0.75, Bicycle 0.75 all feasible
            # Garden 0.75 is already feasible at tau=0.020
            # So need Room 0.75 AND Bicycle 0.75
            if scene == "room":
                falsification = (
                    "HIGH — Room 0.75 is necessary (with Bicycle 0.75) to make scale 0.75 globally feasible. "
                    "If 0.75 becomes globally feasible, best fixed cost drops from 33.15ms to 19.41ms, "
                    "reducing advantage from 17.13ms to 3.39ms (a 13.74ms reduction). "
                    "If Room 0.75 is measured INFEASIBLE, the global-0.75 threat is eliminated entirely."
                )
            elif scene == "bicycle":
                falsification = (
                    "HIGH — Bicycle 0.75 is the other half of the global-0.75 gate. "
                    "If both Room 0.75 and Bicycle 0.75 are feasible, advantage drops to 3.39ms. "
                    "If Bicycle 0.75 is measured INFEASIBLE, the global-0.75 threat is eliminated. "
                    "NOTE: Bicycle 0.75 alone (with Room 0.75 infeasible) actually INCREASES advantage "
                    "from 12.55ms to 17.13ms (oracle gets cheaper, fixed policy unchanged)."
                )
        elif scale == "0.625":
            # 0.625 global feasibility requires all scenes feasible at 0.625
            # Bicycle 0.625 is KNOWN infeasible at tau=0.020 (|dSSIM|=0.0299 > 0.020)
            # So 0.625 can NEVER be globally feasible regardless of Room/Garden 0.625
            falsification = (
                "NONE — Scale 0.625 can never be globally feasible at tau=0.020 because "
                "Bicycle 0.625 is KNOWN infeasible (|dSSIM|=0.0299 > 0.020). "
                "Measuring this variable cannot make a cheaper fixed policy feasible."
            )

        voi["falsification_value"] = falsification

        result["variables"][var] = voi

    save(RESULTS / "b7a_voi_audit.json", result)
    print("\n=== B7-A VOI AUDIT (tau=0.020) ===")
    for var, voi in result["variables"].items():
        print(f"\n  {var}:")
        print(f"    nominal = {voi['nominal']}ms")
        print(f"    max_conditional = {voi['max_conditional_swing']}ms")
        print(f"    falsification = {voi['falsification_value'][:80]}...")
        if voi["max_swing_detail"]:
            d = voi["max_swing_detail"]
            print(f"    max_swing_case: {d['direction']}, swing={d['swing']}ms")

    return result, all_assignments


# ============================================================
# B7-B: Global 0.75 feasibility audit
# ============================================================

def phase_b7b():
    result = {
        "phase": "B7-B",
        "description": "Explicit audit of scale=0.75 global feasibility at tau=0.020",
        "tau": TAU,
        "garden_0.75_status": "FEASIBLE (|dSSIM|=0.0183 <= 0.020)",
        "condition": "Scale 0.75 becomes globally feasible iff Room 0.75 is feasible AND Bicycle 0.75 is feasible",
        "combinations": [],
    }

    combos = [
        {"room_0.75": False, "bicycle_0.75": False, "label": "Room75 FAIL, Bicycle75 FAIL"},
        {"room_0.75": True,  "bicycle_0.75": False, "label": "Room75 PASS, Bicycle75 FAIL"},
        {"room_0.75": False, "bicycle_0.75": True,  "label": "Room75 FAIL, Bicycle75 PASS"},
        {"room_0.75": True,  "bicycle_0.75": True,  "label": "Room75 PASS, Bicycle75 PASS"},
    ]

    for combo in combos:
        # Build full assignment (garden_0.625 and room_0.625 remain infeasible = nominal)
        assignment = {
            "room_0.75": combo["room_0.75"],
            "room_0.625": False,
            "garden_0.625": False,
            "bicycle_0.75": combo["bicycle_0.75"],
        }
        r = compute_advantage(assignment)
        combo_result = {
            "label": combo["label"],
            "room_0.75": "PASS" if combo["room_0.75"] else "FAIL",
            "bicycle_0.75": "PASS" if combo["bicycle_0.75"] else "FAIL",
            "global_0.75_feasible": "0.75" in r["global_feasible"],
            "best_fixed_cost_ms": r["best_fixed_cost"],
            "best_fixed_scale": r["best_fixed_scale"],
            "oracle_scales": r["oracle_scales"],
            "oracle_avg_cost_ms": r["oracle_avg_cost"],
            "advantage_ms": r["advantage"],
            "interpretation": "",
        }

        if combo_result["global_0.75_feasible"]:
            combo_result["interpretation"] = (
                "Scale 0.75 IS globally feasible. Fixed cost drops to 19.41ms. "
                "Advantage drops to 3.39ms — still positive but much smaller. "
                "This is the WORST case for adaptive necessity."
            )
        elif combo["bicycle_0.75"]:
            combo_result["interpretation"] = (
                "Bicycle 0.75 feasible but Room 0.75 infeasible. "
                "Scale 0.75 NOT globally feasible. "
                "Bicycle oracle gets cheaper (0.75 instead of 1.0), INCREASING advantage to 17.13ms. "
                "This is ANTI-falsification — adaptive becomes MORE necessary."
            )
        elif combo["room_0.75"]:
            combo_result["interpretation"] = (
                "Room 0.75 feasible but Bicycle 0.75 infeasible. "
                "Scale 0.75 NOT globally feasible. "
                "Room oracle unchanged (0.5 is cheaper than 0.75). "
                "Advantage unchanged at 12.55ms. No effect."
            )
        else:
            combo_result["interpretation"] = (
                "Both infeasible. Nominal case. Advantage = 12.55ms."
            )

        result["combinations"].append(combo_result)

    result["worst_case_advantage"] = min(c["advantage_ms"] for c in result["combinations"] if c["advantage_ms"] is not None)
    result["worst_case_condition"] = "Room75 PASS, Bicycle75 PASS"

    save(RESULTS / "b7b_global_075_audit.json", result)
    print("\n=== B7-B GLOBAL 0.75 AUDIT ===")
    for c in result["combinations"]:
        print(f"  {c['label']}: global_075={c['global_0.75_feasible']}, "
              f"fixed={c['best_fixed_cost_ms']}ms, oracle_avg={c['oracle_avg_cost_ms']}ms, "
              f"advantage={c['advantage_ms']}ms")
    print(f"\n  Worst case advantage: {result['worst_case_advantage']}ms ({result['worst_case_condition']})")
    return result


# ============================================================
# B7-C: Garden 0.625 arithmetic check
# ============================================================

def phase_b7c():
    result = {
        "phase": "B7-C",
        "description": "Verify Garden 0.625 arithmetic at tau=0.020",
        "tau": TAU,
        "computation": {
            "garden_oracle_before": "0.75 (19.41ms) — Garden 0.75 is feasible at tau=0.020",
            "garden_oracle_after": "0.625 (13.92ms) — if Garden 0.625 is feasible",
            "per_scene_saving_ms": 19.41 - 13.92,
            "average_saving_ms": round((19.41 - 13.92) / 3, 2),
            "fixed_feasible_unchanged": "Scale 0.625 cannot be globally feasible (Bicycle 0.625 is KNOWN infeasible at tau=0.020)",
            "advantage_change": round((19.41 - 13.92) / 3, 2),
        },
        "previous_claim": {
            "value": "6.41ms",
            "source": "B6-B value of information ranking",
            "explanation": (
                "The 6.41ms was computed as max(advantage|G625=feasible) - max(advantage|G625=infeasible) "
                "across ALL tau values. The maximum occurred at tau=0.010, NOT tau=0.020. "
                "At tau=0.010, Garden 0.75 is INFEASIBLE (|dSSIM|=0.0183 > 0.010), so Garden 0.625 changes "
                "the Garden oracle from 1.0 (33.15ms) to 0.625 (13.92ms), saving 19.23ms/scene = 6.41ms average. "
                "At tau=0.020, Garden 0.75 IS feasible, so Garden 0.625 only changes from 0.75 (19.41ms) to "
                "0.625 (13.92ms), saving 5.49ms/scene = 1.83ms average."
            ),
            "verdict": "The 6.41ms is CORRECT for tau=0.010 but MISLEADING for tau=0.020. The B6-B ranking "
                       "mixed tau values, reporting the max across all taus. At tau=0.020, the correct swing is 1.83ms.",
        },
        "correct_swing_at_tau_020": 1.83,
        "correct_swing_at_tau_010": 6.41,
        "correction": (
            "The B6-B ranking reported garden_0.625 as highest value (6.41ms) because it took the max swing "
            "across all taus. At tau=0.020 (the max-opportunity tau), the correct swing is only 1.83ms. "
            "Garden 0.625 has NO falsification value at any tau — it only makes the oracle cheaper, never "
            "the fixed policy. The B6-B ranking was not an arithmetic bug but a DEFINITION issue: it ranked "
            "by max swing across taus rather than by falsification value at the target tau."
        ),
    }

    save(RESULTS / "b7c_garden_0625_check.json", result)
    print("\n=== B7-C GARDEN 0.625 ARITHMETIC CHECK ===")
    print(f"  At tau=0.020: swing = {result['correct_swing_at_tau_020']}ms (NOT 6.41ms)")
    print(f"  At tau=0.010: swing = {result['correct_swing_at_tau_010']}ms (where 6.41ms came from)")
    print(f"  Verdict: {result['previous_claim']['verdict'][:80]}...")
    return result


# ============================================================
# B7-D: Rank training experiments
# ============================================================

def phase_b7d(b7a, b7b):
    result = {
        "phase": "B7-D",
        "description": "Rank missing-scale training experiments by multiple criteria",
        "experiments": {},
    }

    experiments = {
        "bicycle_0.75": {
            "scene": "bicycle",
            "scale": "0.75",
            "falsification_ability": b7a["variables"]["bicycle_0.75"]["falsification_value"][:100],
            "max_conditional_swing_ms": b7a["variables"]["bicycle_0.75"]["max_conditional_swing"],
            "nominal_effect_ms": b7a["variables"]["bicycle_0.75"]["nominal"],
            "swing_direction": "BOTH — increases advantage when R75 infeasible, decreases when R75 feasible",
            "training_cost": "20K remaining iterations at ~49ms/iter = ~16 min GPU",
            "checkpoint_available": "YES — s0.750_iter_10000.pt (695MB, Gaussian state only, NO optimizer state)",
            "resumable": "PARTIAL — checkpoint has Gaussian parameters from correct scale=0.75 trajectory at 10K, but no optimizer state. Resume requires fresh Adam optimizer.",
            "remaining_iterations": 20000,
            "provides_metrics": "PSNR, SSIM, LPIPS at 30K",
        },
        "room_0.75": {
            "scene": "room",
            "scale": "0.75",
            "falsification_ability": b7a["variables"]["room_0.75"]["falsification_value"][:100],
            "max_conditional_swing_ms": b7a["variables"]["room_0.75"]["max_conditional_swing"],
            "nominal_effect_ms": b7a["variables"]["room_0.75"]["nominal"],
            "swing_direction": "ZERO when B75 infeasible, DECREASES advantage by 13.74ms when B75 feasible",
            "training_cost": "30K full run at ~50ms/iter = ~25 min GPU (no checkpoint to resume from)",
            "checkpoint_available": "NO — no Room 0.75 training has ever been run",
            "resumable": "NO — must train from scratch (full 30K)",
            "remaining_iterations": 30000,
            "provides_metrics": "PSNR, SSIM at 30K (LPIPS may need separate evaluation — Room has no LPIPS data)",
        },
        "garden_0.625": {
            "scene": "garden",
            "scale": "0.625",
            "falsification_ability": "NONE — only makes oracle cheaper, cannot make fixed policy cheaper",
            "max_conditional_swing_ms": b7a["variables"]["garden_0.625"]["max_conditional_swing"],
            "nominal_effect_ms": b7a["variables"]["garden_0.625"]["nominal"],
            "swing_direction": "ALWAYS INCREASES advantage by 1.83ms (anti-falsification)",
            "training_cost": "20K remaining at ~42ms/iter = ~14 min GPU",
            "checkpoint_available": "YES — s0.625_iter_10000.pt (Gaussian state only, NO optimizer state)",
            "resumable": "PARTIAL — same limitation as Bicycle 0.75 (no optimizer state)",
            "remaining_iterations": 20000,
            "provides_metrics": "PSNR, SSIM, LPIPS at 30K",
        },
        "room_0.625": {
            "scene": "room",
            "scale": "0.625",
            "falsification_ability": "NONE — 0.625 can never be globally feasible (Bicycle 0.625 KNOWN infeasible)",
            "max_conditional_swing_ms": b7a["variables"]["room_0.625"]["max_conditional_swing"],
            "nominal_effect_ms": b7a["variables"]["room_0.625"]["nominal"],
            "swing_direction": "ZERO — Room 0.5 is cheaper, 0.625 cannot improve Room oracle",
            "training_cost": "30K full run at ~45ms/iter = ~22 min GPU",
            "checkpoint_available": "NO — no Room 0.625 training has ever been run",
            "resumable": "NO — must train from scratch",
            "remaining_iterations": 30000,
            "provides_metrics": "PSNR, SSIM at 30K (no LPIPS for Room)",
        },
    }

    # Ranking
    result["experiments"] = experiments
    result["ranking"] = [
        {
            "rank": 1,
            "experiment": "bicycle_0.75",
            "justification": (
                "HIGH falsification value (half of global-0.75 gate), "
                "PARTIAL resumability (10K checkpoint exists), "
                "LOW remaining cost (20K iters, ~16 min), "
                "provides LPIPS at 30K. "
                "If measured INFEASIBLE, global-0.75 threat eliminated. "
                "If measured FEASIBLE, need Room 0.75 to complete the gate."
            ),
        },
        {
            "rank": 2,
            "experiment": "room_0.75",
            "justification": (
                "HIGH falsification value (other half of global-0.75 gate), "
                "HIGHEST max conditional swing (13.74ms), "
                "but NO checkpoint (full 30K from scratch, ~25 min), "
                "NO LPIPS for Room. "
                "Should be measured AFTER Bicycle 0.75 if B75 is feasible."
            ),
        },
        {
            "rank": 3,
            "experiment": "garden_0.625",
            "justification": (
                "NO falsification value (only helps oracle, not fixed policy), "
                "low swing (1.83ms at tau=0.020), "
                "partial resumability. "
                "Should be prioritized only AFTER global-0.75 threat is resolved."
            ),
        },
        {
            "rank": 4,
            "experiment": "room_0.625",
            "justification": (
                "NO falsification value, ZERO swing, no checkpoint, full 30K needed. "
                "Lowest priority."
            ),
        },
    ]

    save(RESULTS / "b7d_experiment_ranking.json", result)
    print("\n=== B7-D EXPERIMENT RANKING ===")
    for r in result["ranking"]:
        print(f"  Rank {r['rank']}: {r['experiment']} — {r['justification'][:80]}...")
    return result


# ============================================================
# B7-E: Missing metric acquisition feasibility
# ============================================================

def phase_b7e():
    result = {
        "phase": "B7-E",
        "description": "Feasibility of obtaining missing metrics cheaply",
        "checkpoints": {},
    }

    result["checkpoints"] = {
        "bicycle_0.75": {
            "existing_10K_checkpoint": "YES — results/reference_v1/s23/bicycle/checkpoints/s0.750_iter_10000.pt",
            "checkpoint_size_bytes": 695307289,
            "checkpoint_contents": "Gaussian parameters only (xyz, shs, scaling, rotation, opacity, max_radii2D, xyz_gradient_accum, denom, spatial_lr_scale, num_points)",
            "optimizer_state_preserved": False,
            "N_gaussians_at_10K": 2803644,
            "correct_fixed_scale_trajectory": "YES — checkpoint is from the scale=0.75 screening run (10K iterations at C42 scale=0.75)",
            "can_resume": "PARTIAL — can load Gaussian state and continue training with fresh optimizer. NOT identical to continuous 30K run because Adam momentum is lost. After 10K, densification is mostly complete, so impact should be limited.",
            "remaining_iterations": 20000,
            "estimated_time_min": 16,
            "iteration_time_ms": 49.09,
            "caveat": "Resume with fresh optimizer means the trajectory diverges from a continuous 30K run. The 10K Gaussian state is correct, but optimizer momentum rebuilds from zero. Scientific validity depends on whether post-10K optimization is dominated by Gaussian state (likely) or optimizer momentum (unlikely after densification).",
        },
        "room_0.75": {
            "existing_10K_checkpoint": "NO — Room has never been trained at scale 0.75",
            "optimizer_state_preserved": False,
            "correct_fixed_scale_trajectory": "NO — no scale=0.75 trajectory exists for Room",
            "can_resume": "NO — must train full 30K from scratch",
            "remaining_iterations": 30000,
            "estimated_time_min": 25,
            "caveat": "Full 30K from scratch. Room is a small scene (952K Gaussians at scale 1.0), so training is relatively fast.",
        },
        "garden_0.625": {
            "existing_10K_checkpoint": "YES — results/reference_v1/s23/garden/checkpoints/s0.625_iter_10000.pt",
            "checkpoint_contents": "Gaussian parameters only (same structure as Bicycle 0.75)",
            "optimizer_state_preserved": False,
            "N_gaussians_at_10K": 2516004,
            "correct_fixed_scale_trajectory": "YES — checkpoint is from the scale=0.625 screening run",
            "can_resume": "PARTIAL — same limitation as Bicycle 0.75 (no optimizer state)",
            "remaining_iterations": 20000,
            "estimated_time_min": 14,
            "iteration_time_ms": 41.9,
            "caveat": "Same optimizer state limitation. But Garden 0.625 has NO falsification value, so this should not be prioritized.",
        },
        "room_0.625": {
            "existing_10K_checkpoint": "NO — Room has never been trained at scale 0.625",
            "correct_fixed_scale_trajectory": "NO",
            "can_resume": "NO — full 30K from scratch",
            "remaining_iterations": 30000,
            "estimated_time_min": 22,
            "caveat": "Full 30K needed. Zero falsification value. Lowest priority.",
        },
    }

    result["key_finding"] = (
        "Bicycle 0.75 is the ONLY experiment with both (1) high falsification value and (2) an existing "
        "checkpoint from the correct fixed-scale trajectory. The checkpoint lacks optimizer state, making "
        "it a PARTIAL resume, but the Gaussian state at 10K is from the correct scale=0.75 run. "
        "Room 0.75 has higher max conditional swing (13.74ms vs 9.16ms) but requires full 30K from scratch. "
        "Garden 0.625 and Room 0.625 have no falsification value."
    )

    result["resume_trajectory_validity"] = (
        "A 30K final-quality point from a fixed-scale method requires the correct fixed-scale training trajectory. "
        "A one-shot evaluation of a checkpoint trained under another scale does NOT fill this UNKNOWN. "
        "The Bicycle 0.75 checkpoint IS from the correct scale=0.75 trajectory (10K iterations), so resuming "
        "is scientifically valid IF the fresh optimizer does not materially change the post-10K trajectory. "
        "This assumption is reasonable for 3DGS (densification mostly complete by 10K) but not formally proven."
    )

    save(RESULTS / "b7e_acquisition_feasibility.json", result)
    print("\n=== B7-E ACQUISITION FEASIBILITY ===")
    for name, ckpt in result["checkpoints"].items():
        print(f"  {name}: checkpoint={'YES' if 'YES' in str(ckpt.get('existing_10K_checkpoint','')) else 'NO'}, "
              f"resumable={ckpt.get('can_resume','?')[:10]}, "
              f"remaining={ckpt.get('remaining_iterations','?')} iters")
    return result


# ============================================================
# FINAL GATE
# ============================================================

def final_gate(b7a, b7b, b7c, b7d, b7e):
    gate = {
        "experiment": "B7_VOI_AUDIT",
        "date": "2026-09-15",
    }

    # Determine if the previous 6.41ms was correct or incorrect
    previous_correct = "The 6.41ms is CORRECT for tau=0.010 but MISLEADING for tau=0.020. Not a bug — a definition issue."

    # Best single experiment
    best_exp = "bicycle_0.75"

    gate.update({
        "B7_VOI_AUDIT": "PASS",
        "VOI": {
            "room_075": {
                "nominal": b7a["variables"]["room_0.75"]["nominal"],
                "max_conditional": b7a["variables"]["room_0.75"]["max_conditional_swing"],
                "falsification_value": "HIGH — necessary for global-0.75 feasibility (with B75)",
            },
            "room_0625": {
                "nominal": b7a["variables"]["room_0.625"]["nominal"],
                "max_conditional": b7a["variables"]["room_0.625"]["max_conditional_swing"],
                "falsification_value": "NONE — 0.625 cannot be globally feasible (Bicycle 0.625 KNOWN infeasible)",
            },
            "garden_0625": {
                "nominal": b7a["variables"]["garden_0.625"]["nominal"],
                "max_conditional": b7a["variables"]["garden_0.625"]["max_conditional_swing"],
                "falsification_value": "NONE — only makes oracle cheaper, never fixed policy",
            },
            "bicycle_075": {
                "nominal": b7a["variables"]["bicycle_0.75"]["nominal"],
                "max_conditional": b7a["variables"]["bicycle_0.75"]["max_conditional_swing"],
                "falsification_value": "HIGH — other half of global-0.75 gate",
            },
        },
        "GLOBAL_075": {
            "condition": "Room 0.75 AND Bicycle 0.75 both feasible (Garden 0.75 already feasible)",
            "if_feasible_fixed_cost": 19.41,
            "worst_adaptive_advantage": b7b["worst_case_advantage"],
            "worst_case": "Room75 PASS, Bicycle75 PASS -> advantage = 3.39ms (still positive but 73% reduction)",
        },
        "PREVIOUS_GARDEN_0625_6P41": {
            "verdict": "CORRECT for tau=0.010, INCORRECT for tau=0.020",
            "explanation": b7c["previous_claim"]["explanation"],
        },
        "GPU_PRIORITY": {
            "1": "bicycle_0.75 (HIGH falsification, PARTIAL resume, ~16 min, provides LPIPS)",
            "2": "room_0.75 (HIGH falsification, NO resume, ~25 min, no LPIPS)",
            "3": "garden_0.625 (NO falsification, PARTIAL resume, ~14 min)",
            "4": "room_0.625 (NO falsification, NO resume, ~22 min)",
        },
        "BEST_SINGLE_EXPERIMENT": best_exp,
        "ESTIMATED_GPU_COST": "~16 minutes (20K iterations at ~49ms/iter)",
        "RESUMABLE": "PARTIAL — 10K checkpoint exists with Gaussian state from correct scale=0.75 trajectory, but no optimizer state",
        "RUN_AUTHORIZED": "YES",
        "authorization_rationale": (
            "Bicycle 0.75 has HIGH falsification value (half of the global-0.75 gate that could reduce "
            "the adaptive advantage from 12.55ms to 3.39ms) and is the ONLY experiment with an existing "
            "checkpoint from the correct fixed-scale trajectory. The checkpoint lacks optimizer state, "
            "making it a PARTIAL resume, but the Gaussian state at 10K is valid. The remaining 20K "
            "iterations cost ~16 minutes. If Bicycle 0.75 is measured INFEASIBLE, the global-0.75 "
            "threat is eliminated and no further measurement is needed. If FEASIBLE, Room 0.75 becomes "
            "the next priority. Garden 0.625 should be deferred until the global-0.75 threat is resolved."
        ),
        "caveats": [
            "Resume uses fresh optimizer (no optimizer state in checkpoint) — trajectory may differ from continuous 30K run",
            "This is ONE experiment only — no other GPU experiments authorized",
            "If result shows |dSSIM| > 0.020 (INFEASIBLE), global-0.75 threat eliminated, B6 PASS conclusion strengthened",
            "If result shows |dSSIM| <= 0.020 (FEASIBLE), need Room 0.75 to complete the gate",
            "Must not touch Candidate C R3 timing-sensitive work",
        ],
    })

    save(RESULTS / "b7_final_gate.json", gate)
    print("\n" + "=" * 70)
    print("=== B7 FINAL GATE ===")
    print(f"  B7_VOI_AUDIT = {gate['B7_VOI_AUDIT']}")
    print(f"  BEST_SINGLE_EXPERIMENT = {gate['BEST_SINGLE_EXPERIMENT']}")
    print(f"  ESTIMATED_GPU_COST = {gate['ESTIMATED_GPU_COST']}")
    print(f"  RESUMABLE = {gate['RESUMABLE'][:50]}...")
    print(f"  RUN_AUTHORIZED = {gate['RUN_AUTHORIZED']}")
    print(f"\n  Rationale: {gate['authorization_rationale'][:200]}...")
    return gate


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("C42++ B7 — Falsification-First Missing-Scale Gate")
    print("=" * 70)

    b7a, all_assignments = phase_b7a()
    b7b = phase_b7b()
    b7c = phase_b7c()
    b7d = phase_b7d(b7a, b7b)
    b7e = phase_b7e()
    gate = final_gate(b7a, b7b, b7c, b7d, b7e)

    print("\n" + "=" * 70)
    print("B7 ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
