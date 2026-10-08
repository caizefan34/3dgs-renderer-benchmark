#!/usr/bin/env python3
"""H0-FROZEN analyzer (H0_FROZEN_PREREG.md).

Reduced-cell, frozen-checkpoint H0. Runs the ORIGINAL H0 gates unchanged over
the 4 cells whose checkpoints genuinely exist in the frozen C0 trajectory:
    room@15000, room@25000, bicycle@15000, bicycle@25000
The 5K cells (room@5000, bicycle@5000) are ABSENT from the frozen trajectory
(H0_CHECKPOINT_BLOCKER) and are NOT regenerated as canonical (user decision A:
do not call regenerated checkpoints canonical; do not relax the identity gate).
This analyzer therefore NEVER emits a verdict on the original 6-cell H0 grid and
the original h0_analyze.py is NOT reused on this reduced set.

Input 1 (H0_SUMMARY.json): produced by h0_summarize.py over the 4 frozen cells.
    schema: cells[cell]["families"]["K4"|"K8"]["E2D"|"OPA"] = {metrics...}
    A metric is None when the paired NATIVE baseline did not descend on held-out
    (D_native <= 0) for ALL rollouts -> BASE_NON_DESCT; None is a gate FAILURE.
Input 2 (H0_PARITY.json): the frozen endpoint-parity artifact. Per-cell
    parity_pass = e2d_parity_pass AND opa_parity_pass (already PASS for all 4).

Verdict namespace is H0_FROZEN_* so it can NEVER overwrite the original H0_*
verdicts. Gates are the ORIGINAL H0 gates, verbatim (no threshold changes).
The analyzer owns the verdict; no human override.
"""
import json
import sys
from pathlib import Path

CELLS = {"room@15000", "room@25000", "bicycle@15000", "bicycle@25000"}
FAMILIES = ("E2D", "OPA")


def ge(v, t):
    """Gate comparison: None (BASE_NON_DESCT / undefined) is always a FAILURE."""
    return v is not None and v >= t


def strong(x):
    k4, k8 = x["K4"], x["K8"]
    return (
        bool(x["parity_pass"]) and
        ge(k4["p_hold_median"], 1.20) and
        ge(k4["p_hold_p10"], 0.85) and
        ge(k4["e_hold_median"], 1.15) and
        ge(k8["p_hold_median"], 1.30) and
        ge(k8["p_hold_p10"], 1.00) and
        ge(k8["e_hold_median"], 1.20) and
        ge(k8["p_design_median"], 1.20) and
        ge(k8["psnr_delta_median_db"], -0.03) and
        ge(k8["psnr_delta_worst_db"], -0.10) and
        k8["base_non_descent_count"] <= 2 and
        k8["nan_inf_count"] == 0 and
        k4.get("n_rollouts", 8) == 8 and
        k8.get("n_rollouts", 8) == 8
    )


def promote(x):
    k4, k8 = x["K4"], x["K8"]
    return (
        bool(x["parity_pass"]) and
        ge(k4["p_hold_median"], 1.08) and
        ge(k4["p_hold_p10"], 0.70) and
        ge(k4["e_hold_median"], 1.03) and
        ge(k8["p_hold_median"], 1.15) and
        ge(k8["p_hold_p10"], 0.80) and
        ge(k8["e_hold_median"], 1.08) and
        ge(k8["p_design_median"], 1.10) and
        ge(k8["psnr_delta_median_db"], -0.05) and
        ge(k8["psnr_delta_worst_db"], -0.15) and
        k8["base_non_descent_count"] <= 2 and
        k8["nan_inf_count"] == 0 and
        k4.get("n_rollouts", 8) == 8 and
        k8.get("n_rollouts", 8) == 8
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: h0_frozen_analyze.py H0_SUMMARY.json H0_PARITY.json")
    summ = json.loads(Path(sys.argv[1]).read_text())
    parity = json.loads(Path(sys.argv[2]).read_text())
    cells_in = summ.get("cells", {})
    missing = sorted(CELLS - set(cells_in))
    extra = sorted(set(cells_in) - CELLS)
    if missing or extra:
        out = {"verdict": "H0_FROZEN_INPUT_INVALID",
               "missing_cells": missing, "extra_cells": extra}
    else:
        pmap = {}
        for c, v in parity.get("cells", {}).items():
            pmap[c] = bool(v.get("e2d_parity_pass")) and bool(v.get("opa_parity_pass"))
        detail = {}
        strong_f, promote_f = [], []
        for fam in FAMILIES:
            detail[fam] = {}
            s_all = True
            p_all = True
            for cell in sorted(CELLS):
                fblk = cells_in[cell]["families"]
                x = {"parity_pass": pmap.get(cell, False),
                     "K4": fblk["K4"][fam], "K8": fblk["K8"][fam]}
                s = strong(x)
                q = promote(x)
                detail[fam][cell] = {"strong": s, "promote": q,
                                     "parity_pass": x["parity_pass"],
                                     "K4_p_hold_median": x["K4"]["p_hold_median"],
                                     "K8_p_hold_median": x["K8"]["p_hold_median"],
                                     "K8_base_non_descent": x["K8"]["base_non_descent_count"]}
                s_all &= s
                p_all &= q
            if s_all:
                strong_f.append(fam)
            if p_all:
                promote_f.append(fam)
        if strong_f:
            verdict, selected = "H0_FROZEN_STRONG_PROMOTE", strong_f[0]
        elif promote_f:
            verdict, selected = "H0_FROZEN_PROMOTE", promote_f[0]
        else:
            verdict, selected = "H0_FROZEN_DROP", None
        out = {
            "verdict": verdict,
            "selected_fixed_family": selected,
            "strong_families": strong_f,
            "promote_families": promote_f,
            "parity_pass_by_cell": {c: pmap.get(c, False) for c in sorted(CELLS)},
            "detail": detail,
        }
    outp = Path(sys.argv[1]).with_name("H0_FROZEN_VERDICT.json")
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()