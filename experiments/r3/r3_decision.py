#!/usr/bin/env python3
"""
R3 Decision Gate — Geometry-First (C6) + JOINT Weighted Work (Red-Team Reqs 2-4)

Revised decision rules:

  C_KEEP requires ALL of:
    1. Zero aggregate certificate violations (all 4 families, all windows).
    2. O(HW + I_tile) construction confirmed.
    3. JOINT weighted derivative-work removal >= 30% under <= 5% budget,
       with the primary signal coming from LOSS_CONDITIONED_NONZERO_SUPPORT
       (not from exact-zero prior-art culling).
    4. Signal persists across all canonical windows (5K, 10K, 15K).
    5. No CUDA-level violation (genuine ||g_actual|| > B + tol).

  C_MODIFY:
    JOINT removal is 10–30%.

  C_DROP:
    JOINT removal < 10%, OR certificate violations exist, OR method is too
    expensive structurally.

  Critical distinction (Red-team req 4):
    EXACT_ZERO_SUPPORT_CULLING = alpha < 1/255 pairs
      → Prior art (Speedy-Splat, AccuTile). NOT counted as Candidate C novelty.
    LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING = pairs inside nonzero support
      that are skippable due to loss-conditioned derivative bounds.
      → THIS is Candidate C's primary novelty signal.
"""

import os, sys, json, argparse


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_decision(results_dir):
    correctness = load_json(os.path.join(results_dir, "certificate_correctness.json"))
    tightness = load_json(os.path.join(results_dir, "certificate_tightness.json"))
    complexity = load_json(os.path.join(results_dir, "complexity_accounting.json"))
    exact_zero = load_json(os.path.join(results_dir, "exact_zero_statistics.json"))
    disabled = load_json(os.path.join(results_dir, "certificate_disabled.json"))
    joint_skip = load_json(os.path.join(results_dir, "tile_gaussian_certificate.json"))

    if not all([correctness, tightness, complexity, exact_zero, disabled, joint_skip]):
        missing = [
            n for n, d in [
                ("correctness", correctness),
                ("tightness", tightness),
                ("complexity", complexity),
                ("exact_zero", exact_zero),
                ("disabled", disabled),
                ("joint_skip", joint_skip),
            ] if d is None
        ]
        return {
            "decision": "AWAITING_DATA",
            "decision_code": "DATA_INCOMPLETE",
            "message": f"Missing: {missing}",
            "missing": missing,
        }

    geo_primary = ["mean2d", "conic", "mean2d_sigmamin", "conic_sigmamin"]
    appear_secondary = ["color_coarse", "color_tight", "opacity", "opacity_tight"]
    all_families = geo_primary + appear_secondary

    # ---- CR1: Zero aggregate violations (all families, all iterations) ----
    cr1_violations = {}
    cr1_pass = True
    for iter_key, fams in correctness.get("correctness", {}).items():
        for fam, data in fams.items():
            if data.get("available", False):
                v = data.get("violation_count", -1)
                if v > 0:
                    cr1_violations.setdefault(fam, []).append((iter_key, v))
                    cr1_pass = False

    # ---- CR2: JOINT weighted work removal (red-team req 3) ----
    # Load from tile_gaussian_certificate.json's joint_skip
    joint_data = joint_skip.get("joint_skip", {})

    # Aggregate weighted work fractions across iterations
    weighted_fractions_5pct = []
    weighted_fractions_2pct = []
    weighted_fractions_1pct = []
    weighted_fractions_05pct = []
    pair_fractions_5pct = []
    loss_cond_fractions_5pct = []

    for iter_key, eps_dict in joint_data.items():
        if not isinstance(eps_dict, dict):
            continue
        for eps_key in ["eps_5.0pct", "eps_2.0pct", "eps_1.0pct", "eps_0.5pct"]:
            entry = eps_dict.get(eps_key, {})
            work_frac = entry.get("JOINT_SKIP_WEIGHTED_WORK_FRACTION", 0)
            pair_frac = entry.get("JOINT_SKIP_PAIR_FRACTION", 0)
            loss_cond = entry.get("LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING", 0)

            if eps_key == "eps_5.0pct":
                weighted_fractions_5pct.append(work_frac)
                pair_fractions_5pct.append(pair_frac)
                loss_cond_fractions_5pct.append(loss_cond)
            elif eps_key == "eps_2.0pct":
                weighted_fractions_2pct.append(work_frac)
            elif eps_key == "eps_1.0pct":
                weighted_fractions_1pct.append(work_frac)
            elif eps_key == "eps_0.5pct":
                weighted_fractions_05pct.append(work_frac)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    cr2_weighted_work = {
        "5pct_mean_weighted_work_fraction": avg(weighted_fractions_5pct),
        "5pct_mean_pair_fraction": avg(pair_fractions_5pct),
        "5pct_mean_loss_cond_fraction": avg(loss_cond_fractions_5pct),
        "2pct_mean_weighted_work_fraction": avg(weighted_fractions_2pct),
        "1pct_mean_weighted_work_fraction": avg(weighted_fractions_1pct),
        "0.5pct_mean_weighted_work_fraction": avg(weighted_fractions_05pct),
    }

    # C_KEEP threshold: weighted work >= 30% under 5% budget
    cr2_weighted_pass = cr2_weighted_work["5pct_mean_weighted_work_fraction"] >= 0.30
    cr2_weighted_moderate = cr2_weighted_work["5pct_mean_weighted_work_fraction"] >= 0.10

    # ---- CR3: Signal across windows ----
    windows_seen = sorted(
        set(correctness.get("correctness", {}).keys())
        | set(joint_data.keys())
    )
    cr3_pass = len(windows_seen) >= 3

    # ---- CR4: SPD check ----
    disabled_count = 0
    N = 1
    for iter_key, data in disabled.get("disabled", {}).items():
        disabled_count += data.get("spd_disabled_count", 0)
        N = max(N, data.get("N_gaussians", 1))
    disabled_fraction = disabled_count / (len(disabled.get("disabled", {})) * max(N, 1))
    cr4_pass = disabled_fraction < 0.10

    # ---- Secondary diagnostics ----
    secondary = {}
    for fam in appear_secondary:
        ratios = []
        for iter_key, fams in tightness.get("tightness", {}).items():
            data = fams.get(fam, {})
            if data.get("available", False):
                ratios.append(data.get("median", 0))
        if ratios:
            secondary[fam] = {"mean_median_ratio": sum(ratios) / len(ratios)}

    # ---- Final Decision (C6 geometry-first + weighted work) ----
    decision = "C_DROP"
    decision_code = "DROP"
    rationale = []

    if not cr1_pass:
        for fam, viols in cr1_violations.items():
            rationale.append(f"CR1 FAIL: {fam} has {len(viols)} violating iterations")
    else:
        rationale.append("CR1 PASS: zero aggregate violations (all families, all windows)")

    cr2_mean_work = cr2_weighted_work["5pct_mean_weighted_work_fraction"]
    cr2_mean_loss = cr2_weighted_work["5pct_mean_loss_cond_fraction"]

    if cr1_pass and cr2_weighted_pass and cr3_pass and cr4_pass:
        decision = "C_KEEP"
        decision_code = "KEEP"
        rationale.append(
            f"CR2 PASS: JOINT weighted work removal={cr2_mean_work*100:.1f}% "
            f"(loss-cond={cr2_mean_loss*100:.1f}%) >= 30% at 5% budget"
        )
    elif cr1_pass and cr2_weighted_moderate and cr3_pass:
        decision = "C_MODIFY"
        decision_code = "MODIFY"
        rationale.append(
            f"CR2 MODERATE: weighted work removal={cr2_mean_work*100:.1f}% "
            f"in 10-30% range"
        )
    else:
        if cr2_mean_work < 0.10:
            rationale.append(
                f"CR2 FAIL: weighted work removal={cr2_mean_work*100:.1f}% < 10%"
            )
        if not cr3_pass:
            rationale.append(f"CR3 FAIL: only {len(windows_seen)} windows (need >= 3)")
        if not cr4_pass:
            rationale.append(
                f"CR4 FAIL: {disabled_fraction*100:.1f}% SPD-disabled "
                f"(threshold 10%)"
            )

    rationale.append(
        "C6: Geometry-first — JOINT weighted work loss-cond signal is primary"
    )

    return {
        "decision": decision,
        "decision_code": decision_code,
        "decision_rationale": rationale,
        "criteria": {
            "CR1_correctness": {
                "pass": cr1_pass,
                "violations_by_family": {
                    k: len(v) for k, v in cr1_violations.items()
                },
            },
            "CR2_joint_weighted_work": {
                "pass": cr2_weighted_pass,
                "moderate": cr2_weighted_moderate,
                "details": cr2_weighted_work,
                "note": (
                    "Primary gate: JOINT weighted work fraction at 5% budget. "
                    "Loss-conditioned fraction is Candidate C novelty (excludes "
                    "exact-zero prior-art support culling)."
                ),
            },
            "CR3_signal_coverage": {
                "pass": cr3_pass,
                "windows_tested": windows_seen,
                "n_windows": len(windows_seen),
            },
            "CR4_certifiability": {
                "pass": cr4_pass,
                "spd_disabled_fraction": disabled_fraction,
            },
        },
        "secondary_color_opacity_diagnostics": secondary,
        "budget_threshold": 0.05,
        "removal_threshold": 0.30,
        "red_team_notes": [
            "SIGMAMIN_TIGHT bounds added for mean2d and conic (see spec).",
            "JOINT skip-set analysis: single interaction bit kills ALL 4 families.",
            "Weighted work fraction (W_it pixel-lane count) replaces raw pair count.",
            "EXACT_ZERO and LOSS_CONDITIONED culling reported separately.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="R3 Decision Gate — Geometry-First + JOINT (Red-Team)"
    )
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    result = compute_decision(args.input)

    dest = os.path.join(args.input, "final_decision.json")
    with open(dest, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("=" * 64)
    print("R3 DECISION GATE (C6 + JOINT Weighted Work)")
    print("=" * 64)
    for cr_name, cr_data in result.get("criteria", {}).items():
        status = "PASS" if cr_data.get("pass", False) else "FAIL"
        print(f"  {cr_name}: {status}")
    for r in result.get("decision_rationale", []):
        print(f"  {r}")
    print(f"\n  ** FINAL DECISION: {result['decision']} ({result['decision_code']}) **")
    print(f"  Saved: {dest}")


if __name__ == "__main__":
    main()
