#!/usr/bin/env python3
"""H0 summary aggregator (H0_PREREG.md section 8 + H0_SCHEMA.json).

Turns per-rollout trajectories (H0_ROLLOUTS.json from h0_rollout.py) into the
H0_SUMMARY.json schema consumed by h0_analyze.py (the analyzer owns the verdict).

Pairing: each homotopy rollout (E2D/OPA) at (cell, fold, offset, K) is paired
with the NATIVE rollout at the same (fold, offset, K) — identical start state and
update sequence, differing only in the renderer continuation. Per rollout:
  D_native = L0_hold - LK_hold (native);  D_homo = L0_hold - LK_hold (homotopy)
  P_hold = D_homo / D_native   (only when D_native > 0; else BASE_NON_DEScent)
  P_design analog on the design objective;  E_hold = P_hold * (T_native/T_homo)
  k_cross_8 = earliest j<=8 with homotopy held-out <= native K=8 held-out target
  psnr_delta = homotopy holdout_psnr - native holdout_psnr
Aggregates over the 8 paired rollouts (2 folds x 4 offsets): median / p10 of
P_hold, medians of E_hold/P_design/PSNR-delta, worst PSNR-delta, BASE_NON_DEScent
count, NaN/Inf count.

Decision-independent. Tested here on the room@25K smoke trajectories (NOT the
scientific result); the real run feeds the full 6-cell grid.
"""
import argparse
import json
import math
import os
from statistics import median


def pct10(vals):
    if not vals:
        return None
    s = sorted(vals)
    # linear-interpolation 10th percentile
    if len(s) == 1:
        return s[0]
    idx = 0.1 * (len(s) - 1)
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def load_cell(cell_dir):
    """Load all H0_ROLLOUTS*.json in a cell dir, return list of rollout dicts."""
    rollouts = []
    for fn in sorted(os.listdir(cell_dir)):
        if fn.startswith("H0_ROLLOUTS") and fn.endswith(".json"):
            d = json.load(open(os.path.join(cell_dir, fn)))
            rollouts.extend(d.get("rollouts", []))
    return rollouts


def summarize_cell(rollouts, K):
    """Build the families{E2D,OPA}.K{K} block for one cell."""
    # index native by (fold, offset)
    native = {}
    for r in rollouts:
        if r["arm"] == "NATIVE" and r["K"] == K:
            native[(r["fold"], r["offset"])] = r

    fams = {}
    for fam in ("E2D", "OPA"):
        per = []
        base_non_descent = 0
        nan_inf = 0
        for r in rollouts:
            if r["arm"] != fam or r["K"] != K:
                continue
            key = (r["fold"], r["offset"])
            if key not in native:
                continue
            nat = native[key]
            for t in r["trajectory"]:
                nan_inf += int(t.get("nan_inf", 0))
            L0 = r["trajectory"][0]["native"]["holdout_obj"]
            LK = r["trajectory"][-1]["native"]["holdout_obj"]
            L0d = r["trajectory"][0]["native"]["design_obj"]
            LKd = r["trajectory"][-1]["native"]["design_obj"]
            nL0 = nat["trajectory"][0]["native"]["holdout_obj"]
            nLK = nat["trajectory"][-1]["native"]["holdout_obj"]
            D_homo = L0 - LK
            D_native = nL0 - nLK
            # timing (sum of training-step ms; eval excluded)
            T_homo = sum(r.get("step_times_ms") or []) if r.get("step_times_ms") else None
            T_nat = sum(nat.get("step_times_ms") or []) if nat.get("step_times_ms") else None
            psnr_delta = (r["trajectory"][-1]["native"]["holdout_psnr"]
                          - nat["trajectory"][-1]["native"]["holdout_psnr"])
            rec = {"D_native": D_native, "D_homo": D_homo, "psnr_delta": psnr_delta}
            if D_native > 0:
                rec["P_hold"] = D_homo / D_native
                rec["P_design"] = (L0d - LKd) / (nat["trajectory"][0]["native"]["design_obj"]
                                                 - nat["trajectory"][-1]["native"]["design_obj"]) \
                    if (nat["trajectory"][0]["native"]["design_obj"]
                        - nat["trajectory"][-1]["native"]["design_obj"]) > 0 else None
                if T_homo and T_nat and T_homo > 0:
                    rec["E_hold"] = rec["P_hold"] * (T_nat / T_homo)
                # k_cross_8
                if K == 8:
                    target = nLK
                    kcross = None
                    for t in r["trajectory"][1:]:
                        if t["native"]["holdout_obj"] <= target:
                            kcross = t["step"]; break
                    rec["k_cross_8"] = kcross
                per.append(rec)
            else:
                base_non_descent += 1
                per.append(rec)  # counted, excluded from ratio aggregates
        p_holds = [x["P_hold"] for x in per if x.get("P_hold") is not None]
        p_designs = [x["P_design"] for x in per if x.get("P_design") is not None]
        e_holds = [x["E_hold"] for x in per if x.get("E_hold") is not None]
        psnrs = [x["psnr_delta"] for x in per]
        fams[fam] = {
            "p_hold_median": median(p_holds) if p_holds else None,
            "p_hold_p10": pct10(p_holds) if p_holds else None,
            "e_hold_median": median(e_holds) if e_holds else None,
            "p_design_median": median(p_designs) if p_designs else None,
            "psnr_delta_median_db": median(psnrs) if psnrs else None,
            "psnr_delta_worst_db": min(psnrs) if psnrs else None,
            "base_non_descent_count": base_non_descent,
            "nan_inf_count": nan_inf,
            "n_rollouts": len(per),
        }
    return {"K%d" % K: fams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="comma-separated cell dirs each containing H0_ROLLOUTS*.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells = {}
    for cd in args.cells.split(","):
        cell = os.path.basename(os.path.normpath(cd))
        rollouts = load_cell(cd)
        Ks = sorted({r["K"] for r in rollouts})
        fam_block = {}
        for K in Ks:
            fam_block.update(summarize_cell(rollouts, K))
        cells[cell] = {"families": fam_block}
    summary = {"audit_type": "H0_SUMMARY", "schema_version": "H0-v1", "cells": cells}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
