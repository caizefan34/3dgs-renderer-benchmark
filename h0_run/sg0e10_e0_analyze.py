#!/usr/bin/env python3
"""SG0-E10 E0 analyzer (SG0_E10_PREREG.md section C1).

2-cell cohort: room@10000, bicycle@10000 (canonical 10K exists; 5K absent under
decision A, so SG0-E10 is a STRICT REDUCTION of the original SG0-EARLY 4-cell
grid). Verdict namespace SG0E10_* so it can NEVER overwrite the original SG0_* or
H0_* verdicts.

E0 viability gate (section C1): native Adam K=8 must reduce held-out loss on
>2/8 rollouts (BASE_NON_DESCT <= 2/8) on EVERY cell. If ANY cell fails ->
SPLATMG_CLOSED (stop; do NOT search 3K/7K/12K/other cameras/other K). If ALL
cells pass -> E0_PASS (proceed to the coarse-space oracle, section C2, which is
a SEPARATE later stage with its own gate).

The original sg0_analyze.py is NOT reused (it targets 20K/25K and the oracle).
This analyzer owns the verdict; no human override.
"""
import json
import sys
from pathlib import Path

CELLS = {"room@10000", "bicycle@10000"}
BND_LIMIT = 2  # BASE_NON_DESCT <= 2/8


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: sg0e10_e0_analyze.py SG0E10_E0_SUMMARY.json")
    p = Path(sys.argv[1])
    d = json.loads(p.read_text())
    cells = d.get("cells", {})
    missing = sorted(CELLS - set(cells))
    extra = sorted(set(cells) - CELLS)
    if missing or extra:
        out = {"verdict": "SG0E10_E0_INPUT_INVALID",
               "missing_cells": missing, "extra_cells": extra}
    else:
        detail = {}
        fail_cells = []
        for cell in sorted(CELLS):
            bnd8 = cells[cell]["base_non_descent_K8_count"]
            bnd4 = cells[cell]["base_non_descent_K4_count"]
            ok = (bnd8 <= BND_LIMIT)
            detail[cell] = {"base_non_descent_K8": bnd8, "base_non_descent_K4": bnd4,
                            "e0_pass": bool(ok)}
            if not ok:
                fail_cells.append(cell)
        if fail_cells:
            verdict = "SPLATMG_CLOSED"
        else:
            verdict = "SG0E10_E0_PASS"
        out = {
            "verdict": verdict,
            "failed_cells": fail_cells,
            "e0_pass_by_cell": {c: detail[c]["e0_pass"] for c in sorted(CELLS)},
            "detail": detail,
            "note": ("E0_PASS authorizes the section-C2 coarse-space oracle as a "
                     "SEPARATE preregistered stage; it is NOT run here. SPLATMG_CLOSED "
                     "is terminal for the SplatMG-Early line."),
        }
    outp = p.with_name("SG0E10_E0_VERDICT.json")
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()