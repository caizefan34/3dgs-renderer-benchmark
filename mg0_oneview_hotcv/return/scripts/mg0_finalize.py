#!/usr/bin/env python3
"""MG0 finalize — merge interim outputs into the required deliverable files.

Writes/refreshes:
  MG0_SOURCE_IDENTITY.json, MG0_CHECKPOINT_IDENTITY.json,
  MG0_HOTSET_15K.json, MG0_HOTSET_20K.json (+ bonus MG0_HOTSET_25K.json),
  MG0_VARIANCE_ROOM.json, MG0_VARIANCE_BICYCLE.json, MG0_GROUP_METRICS.json,
  MG0_SPEC_OPERATIONALIZATION.json, MG0_VERDICT.json (once stage 6 is done or
  not needed), MG0_CONTAMINATION.json, MANIFEST_MG0.json.

Idempotent: re-run after stage 6 to refresh the verdict/manifest.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg0_core as C


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def main():
    A = C.ARTIFACTS
    T = os.environ.get("MG0_CONTAM_DIR", os.path.join(C.MG0_ROOT, "logs"))

    # ---------------- identity merges --------------------------------------
    C.json_dump({"campaign": "MG0_OneView_GS_HotCV",
                 "stage": "0_source_checkpoint_audit",
                 "note": "merged per-scene Stage-0 audit records",
                 "scenes": {s: load(f"{A}/MG0_SOURCE_IDENTITY_{s}.json")
                            for s in C.SCENES}},
                f"{A}/MG0_SOURCE_IDENTITY.json")
    C.json_dump({"campaign": "MG0_OneView_GS_HotCV",
                 "note": "merged per-scene checkpoint identity records",
                 "scenes": {s: load(f"{A}/MG0_CHECKPOINT_IDENTITY_{s}.json")
                            for s in C.SCENES}},
                f"{A}/MG0_CHECKPOINT_IDENTITY.json")

    # ---------------- hotsets ----------------------------------------------
    for stage, fname in ((15000, "MG0_HOTSET_15K.json"),
                         (20000, "MG0_HOTSET_20K.json"),
                         (25000, "MG0_HOTSET_25K.json")):
        scenes = {}
        for s in C.SCENES:
            d = load(f"{A}/mg0_hotset_{s}_{stage}.json")
            if d is not None:
                scenes[s] = d
        if scenes:
            C.json_dump({"campaign": "MG0_OneView_GS_HotCV", "source_checkpoint": stage,
                         "note": "frozen nested hot sets; IDs are Gaussian row indices "
                                 "in the row layout; ranking from the fp64 score "
                                 "(MG0_SCORE_<scene>_<stage>.npy)",
                         "scenes": scenes}, f"{A}/{fname}")

    # ---------------- variance / group metrics -----------------------------
    for s in C.SCENES:
        ckpts = {}
        for stage in (20000, 25000):
            raw = load(f"{A}/mg0_variance_{s}_{stage}_raw.json")
            if raw is None:
                continue
            m = raw["metrics"]
            ckpts[str(stage)] = {
                "N_rows": raw["N_rows"], "n_views": raw["n_views"],
                "views": raw["views"], "G_norm": raw["G_norm"],
                "loss_mean": raw["loss_mean"], "view_fused_losses": raw["view_fused_losses"],
                "timing": raw["timing"], "memory_gb_max": raw["memory_gb_max"],
                "metrics_source": raw["metrics_SOURCE"],
                "raw_block": m["raw"],
                "per_K": {str(K): m[str(K)] for K in C.K_LIST},
            }
        if ckpts:
            C.json_dump({"campaign": "MG0_OneView_GS_HotCV", "scene": s,
                         "row_layout": ["xyz3", "scale3", "quat4", "opacity1",
                                        "sh_dc3", "sh_rest45"],
                         "preconditioner": "P_t(x) = x/(sqrt(v_t)+eps) with the "
                                           "per-optimizer-group eps of checkpoint t",
                         "stale_lag_iterations": C.N_STALE_LAG,
                         "note": "20K estimators use 15K controls; 25K estimators "
                                 "use 20K controls; no K/threshold change",
                         "checkpoints": ckpts},
                        f"{A}/MG0_VARIANCE_{s.upper()}.json")

    group_rows = []
    for s in C.SCENES:
        for stage in (20000, 25000):
            raw = load(f"{A}/mg0_variance_{s}_{stage}_raw.json")
            if raw is None:
                continue
            for K in C.K_LIST:
                for grp, _, _ in C.GROUPS:
                    rec = raw["metrics"][str(K)]["per_group"][grp]
                    group_rows.append({"scene": s, "checkpoint": stage, "K": K,
                                       "group": grp, **rec})
    C.json_dump({"campaign": "MG0_OneView_GS_HotCV",
                 "note": "per-parameter-group estimator metrics, all cells; the "
                         "'raw' reference row uses the same P_t preconditioner at "
                         "the evaluation checkpoint",
                 "groups": [g for g, _, _ in C.GROUPS],
                 "rows": group_rows}, f"{A}/MG0_GROUP_METRICS.json")

    # ---------------- operationalization record ----------------------------
    C.json_dump({
        "campaign": "MG0_OneView_GS_HotCV",
        "note": "mechanical operationalizations applied where the prereg leaves "
                "an implementation degree of freedom; no threshold was changed",
        "items": {
            "row_layout": "xyz(3)|scale(3)|quat(4)|opacity(1)|sh_dc(3)|sh_rest(45), "
                          "cols 0:3,3:6,6:10,10:11,11:14,14:59",
            "P_denominator": "sqrt(v_t)+eps computed once per checkpoint elementwise; "
                             "eps read from the checkpoint optimizer param_groups "
                             "(1e-15); SH group eps applies to sh_dc and sh_rest",
            "score": "two full passes in fp64: pass1 mean_v P(g_v); pass2 sum_i "
                     "(P(g_i)-mean)^2; score_j = (sum over d)/(64); stable descending "
                     "argsort (ties by row index)",
            "cbar": "FP32 exact mean of the stored FP16 values (fp64 accumulation "
                    "then fp32 cast); estimator subtracts exactly the stored FP16 "
                    "c_i (cast fp32) and adds c_bar",
            "positive_dot_not_decreased": "global positive-dot fraction (dot(v_i,G)>0) "
                                          ">= raw fraction (dot(g_i,G)>0)",
            "median_gain_beyond_noise": "median_a(delta_b2 - delta_b1) > max over the "
                                        "noise subset anchors of (max-min) of the "
                                        "three B1 DeltaHeldoutLoss repeats",
            "no_catastrophic_group_instability": "params finite after every step in "
                                                 "all anchors AND for every group the "
                                                 "median ||dtheta||_B2 <= 3.0x the "
                                                 "median ||dtheta||_B1",
            "noise_subset": "anchors[0::4] (8 of 32 = 25%)",
            "anchors_and_eval": "anchors_32 = quantile(64->32) of the frozen camera "
                                "list (odd positions); internal_eval_16 = quantile "
                                "(32->16) of the remaining even positions; disjoint; "
                                "frozen in MG0_CAMERA_SET.json before gradients",
            "25K_stage": "no new control store at 25K (no 30K consumer exists); the "
                         "25K hot ranking is recorded for completeness only",
            "cost_probe": "isolated op sequence as in prereg §7; control source per "
                          "checkpoint: 15K->15K, 20K->15K, 25K->20K; normalized vs "
                          "N1_ACTION_TIMING_{scene}.json '{ckpt}_A1' total_wall_ms.median",
            "estimator_arithmetic": "v = (g - c_i) + c_bar evaluated left-to-right in "
                                    "fp32 exactly as the isolated ops implement it",
        }}, f"{A}/MG0_SPEC_OPERATIONALIZATION.json")

    # ---------------- verdict ----------------------------------------------
    gate = load(f"{A}/MG0_GATE.json")
    verdict = {"campaign": "MG0_OneView_GS_HotCV", "gate": None, "verdict": None,
               "rationale": None, "cells": None}
    if gate is None:
        verdict["verdict"] = "PENDING"
        verdict["rationale"] = "MG0_GATE.json not present yet"
    elif gate["verdict"] == "MG0_NO_USEFUL_VARIANCE_SIGNAL":
        verdict.update({"gate": gate["verdict"], "verdict": "MG0_NO_USEFUL_VARIANCE_SIGNAL",
                        "rationale": "no K <= 32768 passed the actionable gate; "
                                     "STOP OneView-GS; no trainer implemented"})
    else:
        summ = load(f"{A}/mg0_onestep_summary.json")
        verdict.update({"gate": gate["verdict"], "K_star": gate["K_star"]})
        if summ is None:
            verdict["verdict"] = "PENDING"
            verdict["rationale"] = "variance gate passed; stage 6 one-step missing"
        else:
            cells = summ["cells"]
            n_cells = len(cells)
            strong_ok = n_cells == 3 and all(
                c["frac_b2_gt_b1"] >= 0.60 and c["median_positive_beyond_noise"]
                and c["no_catastrophic_group_instability"] for c in cells.values())
            act_ok = n_cells == 3 and all(
                c["frac_b2_gt_b1"] >= 0.55 and c["median_paired_diff"] > 0
                for c in cells.values())
            if gate["verdict"] == "MG0_STRONG_VARIANCE_SIGNAL" and strong_ok:
                verdict["verdict"] = "MG0_STRONG_ONEVIEW_OPPORTUNITY"
                verdict["rationale"] = ("strong variance gate passed AND all three "
                                        "one-step cells satisfy >=60% B2>B1, median "
                                        "paired improvement beyond B1 noise, no "
                                        "catastrophic group instability; authorize "
                                        "MG1_1K_SEQUENTIAL_ROLLOUT")
            elif act_ok:
                verdict["verdict"] = "MG0_ACTIONABLE_ONEVIEW_OPPORTUNITY"
                verdict["rationale"] = ("actionable variance gate passed AND all three "
                                        "one-step cells satisfy >=55% B2>B1 with "
                                        "positive median paired improvement; small MG1 "
                                        "rollout authorized, no paper claim")
            else:
                verdict["verdict"] = "MG0_VARIANCE_ONLY_NO_UPDATE_GAIN"
                verdict["rationale"] = ("variance gate passed but the one-step "
                                        "integration failed the frozen conditions; do "
                                        "not assume Adam can consume the estimator "
                                        "unchanged")
            verdict["cells"] = {k: {"frac_b2_gt_b1": c["frac_b2_gt_b1"],
                                    "median_paired_diff": c["median_paired_diff"],
                                    "b1_noise_level": c["b1_noise_level"],
                                    "median_positive_beyond_noise":
                                        c["median_positive_beyond_noise"],
                                    "no_catastrophic_group_instability":
                                        c["no_catastrophic_group_instability"]}
                                for k, c in cells.items()}
    C.json_dump(verdict, f"{A}/MG0_VERDICT.json")

    # ---------------- contamination ----------------------------------------
    cont = {"campaign": "MG0_OneView_GS_HotCV",
            "note": "GPU state before/after every stage; only GPU 2 "
                    "(bcca92b9) was used; raw CSVs preserved in logs/",
            "files": {}}
    for p in sorted(glob.glob(os.path.join(T, "cont_*.csv"))):
        cont["files"][os.path.basename(p)] = {
            "sha256": C.sha256_file(p),
            "content": open(p, errors="replace").read()[:4000]}
    C.json_dump(cont, f"{A}/MG0_CONTAMINATION.json")

    # ---------------- manifest ---------------------------------------------
    files = []
    for root in (A, C.CONTROLS_DIR, C.LOGS_DIR, os.path.dirname(os.path.abspath(__file__))):
        for p in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isfile(p):
                files.append({"path": os.path.relpath(p, C.MG0_ROOT),
                              "bytes": os.path.getsize(p),
                              "sha256": C.sha256_file(p)})
    C.json_dump({"campaign": "MG0_OneView_GS_HotCV", "phase": "finalize",
                 "root": C.MG0_ROOT, "n_files": len(files), "files": files},
                f"{A}/MANIFEST_MG0.json")
    print(f"[mg0_finalize] done; verdict={verdict['verdict']}", flush=True)


if __name__ == "__main__":
    main()
