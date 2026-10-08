"""P14 O2 aggregation + frozen gates (Sections 11-15).

Reads P14_O2_UNITS/*.json and P14_O1_AGG.json (surviving q).

Frozen gates (prereg, per q, over the 6 cells, H=32 measurement):
  O2_PASS(q): median R_stream <= 0.30 AND median R_rebuild <= 0.15 AND
      exact FP32 cache <= 24GB for every cluster (max over clusters).
  O2_HARD_DROP: for every eligible q: median R_stream > 0.50, OR
      median R_rebuild > 0.40, OR >=3/6 cells cache > 40GB
      (family-failure reading; each condition family-wide).
If no q survives O1, only q=0.20 diagnostic O2 is gate-eligible.
R_stream gate uses the conservative max(pre, post) per cell; both are
reported. R_rebuild comes from the F(q,32) pre->post comparison.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C


def main():
    o1 = json.load(open(os.path.join(C.P14_ROOT, "artifacts",
                                     "P14_O1_AGG.json")))
    hard_drop = o1.get("decision") == "O1_HARD_DROP"
    surviving = o1.get("surviving_q", [])
    if hard_drop:
        eligible = []           # O1 hard drop: no O2 at all
        diag_only = []
    elif surviving:
        eligible = list(surviving)
        diag_only = []
    else:
        eligible = []           # no surviving q: diagnostic q=0.20 only
        diag_only = [0.20]

    units = []
    for p in sorted(glob.glob(os.path.join(C.P14_ROOT, "P14_O2_UNITS",
                                           "*.json"))):
        u = json.load(open(p))
        u["_path"] = p
        units.append(u)

    def stats_for(q, H):
        rows = [u for u in units
                if abs(u["q"] - q) < 1e-9 and u["H"] == H]
        per_cell = {}
        for u in rows:
            per_cell[f"{u['scene']}/c{u['cell']}"] = {
                "R_stream_pre": u["R_stream_pre"],
                "R_stream_post": u["R_stream_post"],
                "R_stream_max": u["R_stream_max"],
                "R_rebuild": u["rebuild"]["R_rebuild"],
                "cache_gb_fp32": u["cache_gb_fp32_max"],
                "frozen_bitwise_unchanged":
                    u["frozen_bitwise_unchanged"],
                "totals_pre": u["pre"], "totals_post": u["post"],
            }
        if not per_cell:
            return None
        rs = [v["R_stream_max"] for v in per_cell.values()]
        rb = [v["R_rebuild"] for v in per_cell.values()]
        cache = [v["cache_gb_fp32"] for v in per_cell.values()]
        return {
            "q": q, "H": H, "n_cells": len(per_cell),
            "median_R_stream_max": float(np.median(rs)),
            "median_R_rebuild": float(np.median(rb)),
            "max_cache_gb_fp32": float(np.max(cache)),
            "n_cells_cache_gt_24gb": int(sum(1 for c in cache
                                             if c > 24.0)),
            "n_cells_cache_gt_40gb": int(sum(1 for c in cache
                                             if c > 40.0)),
            "all_frozen_bitwise": all(
                v["frozen_bitwise_unchanged"] for v in per_cell.values()),
            "per_cell": per_cell,
        }

    agg = {"phase": "P14",
           "eligible_q_from_O1": eligible,
           "diagnostic_only_q": diag_only,
           "o1_hard_drop": hard_drop,
           "interpretation_notes": [
               "R_stream gate metric = max(pre, post) per cell "
               "(conservative); pre = 16K initial state, post = state "
               "after the F(q,32) arm.",
               "Cache = n_frozen_segments(max pre/post) x 4112 B "
               "(256 px x 4 FP32 + 16 B metadata); FP16 reported per "
               "unit but NOT gate-eligible.",
               "Family-failure reading of the hard drop as in O1.",
           ], "by_q": {}}

    for q in sorted(set(eligible + diag_only)):
        st32 = stats_for(q, 32)
        st128 = stats_for(q, 128)
        rec = {"H32": st32, "H128": st128}
        if st32 is not None and (q in eligible):
            rec["O2_PASS"] = bool(
                st32["median_R_stream_max"] <= 0.30
                and st32["median_R_rebuild"] <= 0.15
                and st32["max_cache_gb_fp32"] <= 24.0
                and st32["all_frozen_bitwise"])
            rec["gate_eligible"] = True
        else:
            rec["O2_PASS"] = False
            rec["gate_eligible"] = q in eligible
        agg["by_q"][str(q)] = rec

    # hard drop (family reading)
    gate_q = [q for q in eligible if str(q) in agg["by_q"]
              and agg["by_q"][str(q)]["H32"] is not None]
    if hard_drop:
        o2_hard = None   # already stopped at O1
    elif not gate_q:
        o2_hard = None
    else:
        c1 = all(agg["by_q"][str(q)]["H32"]["median_R_stream_max"] > 0.50
                 for q in gate_q)
        c2 = all(agg["by_q"][str(q)]["H32"]["median_R_rebuild"] > 0.40
                 for q in gate_q)
        c3 = all(agg["by_q"][str(q)]["H32"]["n_cells_cache_gt_40gb"] >= 3
                 for q in gate_q)
        o2_hard = bool(c1 or c2 or c3)
    agg["O2_HARD_DROP"] = o2_hard
    o2_pass_q = [q for q in gate_q
                 if agg["by_q"][str(q)]["O2_PASS"]]
    agg["O2_PASS_q"] = o2_pass_q
    if hard_drop:
        agg["decision"] = "STOPPED_AT_O1 (no O2 gates evaluated)"
    elif o2_hard:
        agg["decision"] = "O2_HARD_DROP"
    elif o2_pass_q:
        agg["decision"] = "O2_CANDIDATE"
    else:
        agg["decision"] = ("O2_NO_CANDIDATE"
                           + (" (diagnostic only)" if diag_only else ""))
    agg["next"] = ("STOP_P14" if (hard_drop or o2_hard)
                   else ("O3 for q in O1_PASS ∩ O2_PASS = "
                         f"{sorted(set(o1.get('surviving_q', [])) & set(o2_pass_q))}"
                         if o2_pass_q else
                         "no q passes O2; O3 not allowed -> STOP_P14"))
    C.jwrite(os.path.join(C.P14_ROOT, "artifacts", "P14_O2_AGG.json"), agg)
    print(json.dumps({k: agg[k] for k in
                      ("decision", "O2_PASS_q", "next")}, indent=2))


if __name__ == "__main__":
    main()
