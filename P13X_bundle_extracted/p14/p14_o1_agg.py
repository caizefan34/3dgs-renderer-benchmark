"""P14 O1 aggregation + frozen gates (Sections 9, 10).

Reads P14_O1_UNITS/*.json; computes R_train / P_audit / viability;
evaluates O1_CANDIDATE / O1_HARD_DROP / Stage-B allowance / O1_PASS(q).

Frozen gates (prereg):
  locally viable cell: R_train >= 0.80 AND P_audit <= 0.002 AND frozen
      bitwise identical AND no NaN/Inf
  O1_CANDIDATE: some q<=0.25 with >=5/6 cells viable AND median
      R_train >= 0.85 (Stage A, H=32)
  O1_HARD_DROP: all q<=0.25 median R_train < 0.60 OR (for every q<=0.25)
      >=3/6 cells audit penalty > 1%   [family-failure reading]
  Stage B allowed: some q<=0.25 median R_train (H=32) >= 0.75
  O1_PASS(q): candidate-level thresholds at H=32 AND (if Stage B ran)
      at H=8 and H=128.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C


def load_units():
    units = {}
    for p in sorted(glob.glob(os.path.join(C.P14_ROOT, "P14_O1_UNITS",
                                           "*.json"))):
        u = json.load(open(p))
        if u.get("arm") == "d0":
            units.setdefault("d0", {})[u["scene"]] = u
            continue
        key = (u["scene"], u["cell"], u["arm"], u["q"], u["H"])
        units.setdefault("arms", {})[key] = u
    return units


def cell_tables(units, H):
    """Returns {(scene,cell): {q: rec}} for given H, plus D1 baselines."""
    d1 = {}
    arms = {}
    for (scene, cell, arm, q, h), u in units["arms"].items():
        if h != H:
            continue
        if arm == "d1":
            d1[(scene, cell)] = u
        else:
            arms[(scene, cell, q)] = u
    out = {}
    for (scene, cell, q), u in arms.items():
        d = d1.get((scene, cell))
        rec = {"L_cluster_before": u["L_cluster_before"],
               "L_cluster_after_F": u["L_cluster_after"],
               "L_audit_before": u["L_audit_before"],
               "L_audit_after_F": u["L_audit_after"],
               "freeze_ok": u.get("freeze_ok"),
               "finite_ok": u.get("finite_ok")}
        if d is None:
            rec.update({"R_train": None, "P_audit": None,
                        "G_train_F": u["G_train"], "G_train_D1": None,
                        "viable": False, "reason": "no D1 baseline"})
        else:
            g_f = u["G_train"]
            g_d = d["G_train"]
            rec["G_train_F"] = g_f
            rec["G_train_D1"] = g_d
            rec["L_audit_after_D1"] = d["L_audit_after"]
            rec["L_cluster_after_D1"] = d["L_cluster_after"]
            r_train = (g_f / g_d) if (g_d is not None and g_d > 0) else None
            d_audit = u["L_audit_after"] - d["L_audit_after"]
            p_audit = (d_audit / u["L_audit_before"]
                       if u["L_audit_before"] > 0 else None)
            rec["R_train"] = r_train
            rec["P_audit"] = p_audit
            rec["D_audit"] = d_audit
            ok = (r_train is not None and r_train >= 0.80
                  and p_audit is not None and p_audit <= 0.002
                  and bool(u.get("freeze_ok", False))
                  and bool(u.get("finite_ok", False)))
            rec["viable"] = bool(ok)
        out.setdefault((scene, cell), {})[q] = rec
    return out, d1


def q_stats(table_by_cell, q):
    """Median R_train, viability count, audit>1% count over 6 cells."""
    rs, viable, audit_bad, valid = [], 0, 0, 0
    per_cell = {}
    for (scene, cell), qmap in sorted(table_by_cell.items()):
        rec = qmap.get(q)
        if rec is None:
            per_cell[f"{scene}/c{cell}"] = None
            continue
        per_cell[f"{scene}/c{cell}"] = {
            "R_train": rec["R_train"], "P_audit": rec["P_audit"],
            "viable": rec["viable"]}
        if rec["R_train"] is not None:
            rs.append(rec["R_train"])
            valid += 1
        if rec["viable"]:
            viable += 1
        if rec["P_audit"] is not None and rec["P_audit"] > 0.01:
            audit_bad += 1
    return {
        "q": q, "n_cells": len(table_by_cell), "n_valid_R_train": valid,
        "median_R_train": float(np.median(rs)) if rs else None,
        "n_viable": viable, "n_audit_penalty_gt_1pct": audit_bad,
        "per_cell": per_cell,
    }


def main():
    units = load_units()
    agg = {"phase": "P14", "H_stage_A": C.H_STAGE_A,
           "interpretation_notes": [
               "R_train = G_train^F / G_train^D1 (same cluster schedule, "
               "same initial state); D0 is external reference only.",
               "O1_HARD_DROP family reading: median<0.60 for ALL q<=0.25, "
               "or >=3/6 cells audit>1% for EVERY q<=0.25 (no clean q).",
               "O1_PASS(q) applies the candidate thresholds (>=5/6 viable "
               "AND median R_train>=0.85) at every tested H.",
           ]}

    # ---- Stage A -----------------------------------------------------
    tableA, d1A = cell_tables(units, C.H_STAGE_A)
    agg["stage_A"] = {"per_q": [], "d1_G_train": {
        f"{s}/c{c}": u["G_train"] for (s, c), u in sorted(d1A.items())}}
    for q in C.Q_LIST_STAGE_A:
        agg["stage_A"]["per_q"].append(q_stats(tableA, q))
    med_le25 = {r["q"]: r["median_R_train"]
                for r in agg["stage_A"]["per_q"] if r["q"] <= 0.25}
    viable_le25 = {r["q"]: r for r in agg["stage_A"]["per_q"]
                   if r["q"] <= 0.25}
    stageB_allowed = any(m is not None and m >= 0.75
                         for m in med_le25.values())
    o1_candidate = any(
        (r["n_viable"] >= 5 and r["median_R_train"] is not None
         and r["median_R_train"] >= 0.85)
        for r in viable_le25.values())
    hard1 = all(m is None or m < 0.60 for m in med_le25.values())
    hard2 = all(r["n_audit_penalty_gt_1pct"] >= 3
                for r in viable_le25.values())
    o1_hard_drop = bool(hard1 or hard2)
    agg["stage_A"]["stage_B_allowed"] = bool(stageB_allowed)
    agg["stage_A"]["O1_CANDIDATE"] = bool(o1_candidate)
    agg["stage_A"]["O1_HARD_DROP"] = o1_hard_drop

    # ---- Stage B (if units exist) ------------------------------------
    agg["stage_B"] = {"available": False}
    stageB_units = {k: u for k, u in units["arms"].items()
                    if k[4] in C.H_STAGE_B}
    if stageB_units:
        agg["stage_b_ran"] = True
        agg["stage_B"] = {"available": True, "per_H": {}}
        for H in C.H_STAGE_B:
            tableH, d1H = cell_tables(units, H)
            if not tableH:
                continue
            per_q = [q_stats(tableH, q) for q in C.Q_LIST_LE25]
            agg["stage_B"]["per_H"][str(H)] = {
                "per_q": per_q,
                "d1_G_train": {f"{s}/c{c}": u["G_train"]
                               for (s, c), u in sorted(d1H.items())}}

    # ---- O1_PASS per q ------------------------------------------------
    def pass_at_H(H, q):
        tableH, _ = cell_tables(units, H)
        if not tableH:
            return None
        st = q_stats(tableH, q)
        return bool(st["n_viable"] >= 5 and st["median_R_train"] is not None
                    and st["median_R_train"] >= 0.85)

    o1_pass = {}
    for q in C.Q_LIST_LE25:
        checks = {"H32": pass_at_H(C.H_STAGE_A, q)}
        if stageB_units:
            for H in C.H_STAGE_B:
                checks[f"H{H}"] = pass_at_H(H, q)
        o1_pass[str(q)] = {
            "checks": checks,
            "O1_PASS": all(v is True for v in checks.values()),
        }
    agg["O1_PASS_by_q"] = o1_pass
    surviving = [q for q in C.Q_LIST_LE25 if o1_pass[str(q)]["O1_PASS"]]
    agg["surviving_q"] = surviving
    agg["decision"] = (
        "O1_HARD_DROP" if o1_hard_drop else
        ("O1_CANDIDATE" if o1_candidate else "O1_NO_CANDIDATE"))
    agg["next"] = ("STOP_P14 (O1 hard drop; no stage B / O2 / O3)"
                   if o1_hard_drop else
                   ("continue Stage B then O2 for surviving q"
                    if stageB_allowed else
                    "Stage B NOT allowed (no q<=0.25 median R_train>=0.75);"
                    " diagnostic O2 at q=0.20 only, then stop"))
    C.jwrite(os.path.join(C.P14_ROOT, "artifacts", "P14_O1_AGG.json"), agg)
    print(json.dumps({k: agg[k] for k in
                      ("decision", "surviving_q", "next")}, indent=2))


if __name__ == "__main__":
    main()
