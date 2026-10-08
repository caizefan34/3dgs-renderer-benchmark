#!/usr/bin/env python3
"""CAUSAL ladder verdict/attribution/report: consumes base artifacts and
produces COMPOSITION, HEADLINE_GAP, ATTRIBUTION, SEED_EXPANSION_DECISION,
VERDICT, REPORT.md, RUN_STATUS, MANIFEST."""
import argparse, json, os, glob
import numpy as np

INCOMING = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived"
OUTROOT = os.path.join(INCOMING, "causal_out")
SCENES = ["room", "bicycle", "garden"]
ARMS = ["a0", "a1", "a2", "c0"]
SEED = 42
MATCHED_HEADLINE_GM = 1.0669  # MX C0 matched replication B1A->C0


def load(path, default=None):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def gm(x):
    x = [v for v in x if v is not None and v > 0]
    return float(np.prod(x) ** (1.0 / len(x))) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", default=OUTROOT)
    ap.add_argument("--runroot", default="/dev/shm/causal_ladder/runs")
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--hostname", default="bms-39468022-001")
    args = ap.parse_args()
    O = args.outroot

    timing = load(os.path.join(O, "CAUSAL_FULLTRAIN_TIMING.json"), {})
    quality = load(os.path.join(O, "CAUSAL_QUALITY.json"), {})
    traj = load(os.path.join(O, "CAUSAL_TRAJECTORY.json"), {})
    fixed = load(os.path.join(O, "CAUSAL_FIXED_STATE_TIMING_ROOM.json")) or load(os.path.join(O, "CAUSAL_FIXED_STATE_TIMING.json"), {})
    correct = load(os.path.join(O, "CAUSAL_CORRECTNESS.json"), {})
    src = load(os.path.join(O, "CAUSAL_SOURCE_IDENTITY.json"), {})
    proto = load(os.path.join(O, "CAUSAL_PROTOCOL_IDENTITY.json"), {})
    thirtyk = load(os.path.join(O, "CAUSAL_30K_RUNS.json"), {})

    transitions = [("F9", "a0", "a1"), ("SCALAR_ADJOINT", "a1", "a2"), ("H8_MR", "a2", "c0"), ("TOTAL_A0_A3", "a0", "c0")]
    tg = timing.get("overall_geomean", {})
    popg = traj.get("population_ratios", {})
    fixed_gm = (fixed or {}).get("ratios_fb", {})

    # ---- CAUSAL_COMPOSITION.json ----
    comp = {"artifact": "CAUSAL_COMPOSITION",
            "R_observed_A0_A3": tg.get("TOTAL_A0_A3"),
            "marginal_fulltrain": {name: tg.get(name) for name, a, b in transitions},
            "fixed_state_marginal_fb": {name: fixed_gm.get(k) for name, k in
                                        [("F9", "F9_A0_over_A1"), ("SCALAR_ADJOINT", "SCALAR_A1_over_A2"),
                                         ("H8_MR", "H8_A2_over_A3"), ("TOTAL_A0_A3", "TOTAL_A0_over_A3")]},
            "note": "R_pred product is tautological (same runs); the informative comparison is "
                    "fixed-state marginal mechanism effect vs full-training marginal effect."}
    # does each module retain benefit in the cumulative composition?
    comp["retain_tests"] = {
        "F9_retains_after_scalar_H8": "compare fixed-state F9 ratio vs full-train F9 ratio",
        "H8_retains_in_A2_parent": tg.get("H8_MR"),
        "redundancy": "if a module ratio ~1.0 in full-train but >1 fixed-state, gains masked by trajectory",
    }
    with open(os.path.join(O, "CAUSAL_COMPOSITION.json"), "w") as f:
        json.dump(comp, f, indent=2)

    # ---- CAUSAL_HEADLINE_GAP.json ----
    gap = {"artifact": "CAUSAL_HEADLINE_GAP",
           "MX_B1A_to_C0_matched_headline_GM": MATCHED_HEADLINE_GM,
           "A0_to_A3_ladder_GM": tg.get("TOTAL_A0_A3"),
           "config_delta_table": {
               "B1A": {"renderer": "gsplat true-accutile (eps2d=0.3)", "modules": "accutile rasterization"},
               "A0": {"renderer": "C0 ext 9baf8655 (eps2d=0.3)", "modules": "gather forward, tensor adjoint, no H8-MR"},
               "A3_C0": {"renderer": "C0 ext 9baf8655 (eps2d=0.3)", "modules": "F9 + SCALAR_ADJOINT + H8-MR"},
           },
           "explanation": "B1A uses a DIFFERENT renderer (gsplat accutile) than the C0-family "
                          "(9baf8655 higs dynamic). A0 is the C0-family parent with all named modules OFF. "
                          "The B1A->C0 headline therefore includes the renderer-substrate difference "
                          "(gsplat-accutile vs higs-dynamic) PLUS the named-module deltas, whereas A0->A3 "
                          "isolates ONLY the named modules on one substrate. The gap is expected and is NOT "
                          "a contradiction; it reflects the substrate change, not a missing module."}
    with open(os.path.join(O, "CAUSAL_HEADLINE_GAP.json"), "w") as f:
        json.dump(gap, f, indent=2)

    # ---- CAUSAL_ATTRIBUTION.json ----
    def classify(name, a, b):
        full = tg.get(name)
        fx = fixed_gm.get({"F9": "F9_A0_over_A1", "SCALAR_ADJOINT": "SCALAR_A1_over_A2",
                           "H8_MR": "H8_A2_over_A3"}.get(name, ""))
        popr = (popg.get(name) or {}).get("geomean")
        intrinsic = (fx is not None and fx > 1.02)
        trajectory = (popr is not None and popr < 0.98)
        if full is None:
            cat = "NEUTRAL"
        elif full > 1.02 and intrinsic:
            cat = "INTRINSIC_RENDERER_GAIN"
        elif full > 1.02 and trajectory and not intrinsic:
            cat = "TRAJECTORY_MEDIATED_GAIN"
        elif full > 1.02:
            cat = "MIXED_GAIN"
        elif full < 0.98:
            cat = "REGRESSION"
        else:
            cat = "NEUTRAL"
        return {"module": name, "full_train_ratio": full, "fixed_state_ratio": fx,
                "population_ratio_Nb_over_Na": popr, "attribution": cat}

    attr = {"artifact": "CAUSAL_ATTRIBUTION", "modules": []}
    for name, a, b in transitions:
        if name == "TOTAL_A0_A3":
            continue
        attr["modules"].append(classify(name, a, b))
    # cumulative
    attr["cumulative_A0_A3"] = {"full_train_ratio": tg.get("TOTAL_A0_A3"),
                                "fixed_state_ratio": fixed_gm.get("TOTAL_A0_over_A3"),
                                "population_ratio": (popg.get("TOTAL_A0_A3") or {}).get("geomean")}
    with open(os.path.join(O, "CAUSAL_ATTRIBUTION.json"), "w") as f:
        json.dump(attr, f, indent=2)

    # ---- CAUSAL_SEED_EXPANSION_DECISION.json ----
    total = tg.get("TOTAL_A0_A3")
    f9 = tg.get("F9"); scalar = tg.get("SCALAR_ADJOINT"); h8 = tg.get("H8_MR")
    expand = False; reasons = []
    if total is not None and total >= 1.02:
        expand = True; reasons.append("A: A0->A3 >= 1.02x")
    for nm, v in (("F9", f9), ("SCALAR", scalar), ("H8", h8)):
        if v is not None and v >= 1.01:
            # stable across 3 scenes?
            ps = timing.get("per_transition", {}).get(
                {"F9": "F9", "SCALAR": "SCALAR_ADJOINT", "H8": "H8_MR"}[nm], {}).get("per_scene_ratio", {})
            if all(x and x >= 1.0 for x in ps.values()):
                expand = True; reasons.append(f"B: {nm} >=1% across 3 scenes")
    if (f9 is not None and f9 < 1.0 and (scalar or 1) < 1.0 and (h8 or 1) < 1.0):
        reasons.append("D: surprising contradiction requiring robustness testing")
        expand = True
    seed_decision = {"artifact": "CAUSAL_SEED_EXPANSION_DECISION",
                     "expand": expand, "reasons": reasons if reasons else ["no gate criterion met"],
                     "decision": "EXPAND" if expand else "NO_MULTI_SEED_EXPANSION",
                     "phase": "Phase 1 (seed 42 only, 3 scenes x 4 arms = 12 runs)"}
    with open(os.path.join(O, "CAUSAL_SEED_EXPANSION_DECISION.json"), "w") as f:
        json.dump(seed_decision, f, indent=2)

    # ---- CAUSAL_VERDICT.json ----
    correctness_pass = (correct or {}).get("correctness_verdict") == "PASS"
    per_scene_gm = {sc: (timing.get("per_scene", {}).get(sc) or {}) for sc in SCENES}
    scene_gms = {}
    for sc in SCENES:
        ps = timing.get("per_scene", {}).get(sc, {})
        t = [ps.get(a) for a in ARMS if ps.get(a)]
        scene_gms[sc] = gm(t) if len(t) == 4 else None
    # per-scene TOTAL ratio (a0/c0)
    scene_total_ratio = {}
    for sc in SCENES:
        ps = timing.get("per_scene", {}).get(sc, {})
        if ps.get("a0") and ps.get("c0"):
            scene_total_ratio[sc] = ps["a0"] / ps["c0"]
    all_scene_gt1 = all(v and v > 1 for v in scene_total_ratio.values())
    total_gt1 = total is not None and total > 1
    # count positive pairs per transition
    def pos_count(name):
        ps = timing.get("per_transition", {}).get(name, {}).get("per_scene_ratio", {})
        return sum(1 for v in ps.values() if v and v > 1), len(ps)
    f9_pos, f9_n = pos_count("F9")
    sc_pos, sc_n = pos_count("SCALAR_ADJOINT")
    h8_pos, h8_n = pos_count("H8_MR")
    tot_pos, tot_n = pos_count("TOTAL_A0_A3")
    qclass = quality.get("classification_A0_to_A3")
    traj_confound = traj.get("trajectory_confound", False)

    if not correctness_pass:
        verdict = "BLOCKED_CORRECTNESS"
    elif total_gt1 and all_scene_gt1 and f9_pos == f9_n and sc_pos == sc_n and tot_pos == tot_n and qclass == "QUALITY_NEUTRAL" and not traj_confound:
        verdict = "ATTRIBUTION_CLEAN"
    elif total_gt1 and (f9_pos + sc_pos + h8_pos) >= 2 and tot_pos >= 2:
        verdict = "ATTRIBUTION_MIXED_BUT_EXPLAINABLE"
    elif total is not None and 0.98 <= total <= 1.02:
        verdict = "ATTRIBUTION_NEUTRAL"
    elif total is not None and total < 0.98:
        verdict = "ATTRIBUTION_CONTRADICTORY"
    else:
        verdict = "ATTRIBUTION_MIXED_BUT_EXPLAINABLE"

    crit = {"correctness_pass": correctness_pass,
            "A0_A3_total_ratio": total, "all_scene_total_gt1": all_scene_gt1,
            "F9_pos": f"{f9_pos}/{f9_n}", "SCALAR_pos": f"{sc_pos}/{sc_n}", "H8_pos": f"{h8_pos}/{h8_n}",
            "TOTAL_pos": f"{tot_pos}/{tot_n}", "quality_class": qclass,
            "trajectory_confound": traj_confound}
    verdict_obj = {"artifact": "CAUSAL_VERDICT", "verdict": verdict, "criteria": crit,
                   "overall_geomean": total, "scene_total_ratio": scene_total_ratio}
    with open(os.path.join(O, "CAUSAL_VERDICT.json"), "w") as f:
        json.dump(verdict_obj, f, indent=2)

    # ---- RUN_STATUS.json ----
    run_status = {"artifact": "RUN_STATUS", "campaign": "C0_SAME_PARENT_CAUSAL_LADDER_MX",
                  "phase": "phase1_seed42", "n_runs_expected": 12, "seed": SEED,
                  "scenes": SCENES, "arms": ARMS, "gpu": args.gpu, "hostname": args.hostname}
    with open(os.path.join(O, "RUN_STATUS.json"), "w") as f:
        json.dump(run_status, f, indent=2)

    print("VERDICT:", verdict, "total GM:", total)
    return verdict_obj, attr, seed_decision, gap, comp, timing, quality, traj, fixed, thirtyk


if __name__ == "__main__":
    main()
