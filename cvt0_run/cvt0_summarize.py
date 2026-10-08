#!/usr/bin/env python3
"""Aggregate CVT0_RESULTS.json -> CVT0_SUMMARY.json (CVT0_SCHEMA.json cells).

Median over the 3 panels inside each cell (prereg section 9). The analyzer
(cvt0_analyze.py) owns the verdict; this only builds the schema cells.
"""
import json, os, sys
from statistics import median

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_cvt0"
res = json.load(open(os.path.join(OUT, "CVT0_RESULTS.json")))

def med(vals):
    vals = [v for v in vals if v is not None]
    return median(vals) if vals else None

cells = {}
for cell, cr in res.items():
    panels = list(cr["panels"].values())
    n_panels = len(panels)
    complete = all(p["nan_inf_count"] == 0 for p in panels) and n_panels == 3
    H_oracle_positive_count = sum(1 for p in panels
                                  if p["selectors"]["H_oracle"] > 0)
    cells[cell] = {
        "n_panels": n_panels,
        "T_oracle_median": med([p["selectors"]["T_oracle"] for p in panels]),
        "G_random_median": med([p["metrics"]["G_random"] for p in panels]),
        "G_hard_median": med([p["metrics"]["G_hard"] for p in panels]),
        "G_true_median": med([p["metrics"]["G_true"] for p in panels]),
        "H_oracle_positive_count": H_oracle_positive_count,
        "F_pos_median": med([p["metrics"]["F_pos"] for p in panels]),
        "rho_TH_median": med([p["metrics"]["rho_TH"] for p in panels]),
        "complete": bool(complete),
        "nan_inf_count": int(sum(p["nan_inf_count"] for p in panels)),
    }

summary = {
    "schema_version": "CVT0-v1",
    "cells": cells,
    "aggregates": {
        "n_cells": len(cells),
        "n_cells_complete": sum(1 for c in cells.values() if c["complete"]),
    },
}
with open(os.path.join(OUT, "CVT0_SUMMARY.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
