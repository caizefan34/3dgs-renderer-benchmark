#!/usr/bin/env python3
"""MG0 gate — mechanical evaluation of prereg §8 Booleans (no reinterpretation).

STRONG (MG0_STRONG_VARIANCE_SIGNAL): exists one K <= 32768 satisfying for BOTH
  scenes at BOTH 20K and 25K:
    adam_var_ratio <= 0.50
    median cosine gain >= +0.12
    positive-dot fraction does not decrease
    projected full-camera control memory <= 1.5 GiB
    isolated control maintenance <= 10% of native A1 iteration wall
ACTIONABLE: same with <= 0.70 / >= +0.07 / <= 1.5 GiB / <= 15%.
Smallest passing K is selected for the one-step test.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg0_core as C

CELLS = (("room", 20000), ("room", 25000), ("bicycle", 20000), ("bicycle", 25000))


def main():
    v = {}
    for s in C.SCENES:
        p = os.path.join(C.ARTIFACTS, f"MG0_VARIANCE_{s.upper()}.json")
        v[s] = json.load(open(p))
    cost = json.load(open(os.path.join(C.ARTIFACTS, "MG0_COST.json")))
    mem = json.load(open(os.path.join(C.ARTIFACTS, "MG0_MEMORY.json")))

    per_K = {}
    for K in C.K_LIST:
        cells = {}
        for scene, stage in CELLS:
            m = v[scene]["checkpoints"][str(stage)]["per_K"][str(K)]
            c = cost["cells"][f"{scene}|{stage}|{K}"]
            proj = mem["projected_table_memory"][scene]["K"][str(K)]["total_gib"]
            cell = {
                "adam_var_ratio": m["adam_var_ratio"],
                "median_cosine_gain": m["median_cosine_gain"],
                "p10_cosine_gain": m["p10_cosine_gain"],
                "positive_dot_fraction": m["positive_dot_fraction"],
                "positive_dot_fraction_raw": m["positive_dot_fraction_raw"],
                "positive_dot_not_decreased": m["positive_dot_fraction_not_decreased"],
                "projected_memory_gib": proj,
                "control_maintenance_ratio": c["control_maintenance_ratio"],
                "a1_median_ms": c["a1_median_ms"],
                "event_median_ms": c["event_ms"]["median"],
            }
            cell["strong_ok"] = bool(
                cell["adam_var_ratio"] <= 0.50 and cell["median_cosine_gain"] >= 0.12
                and cell["positive_dot_not_decreased"]
                and cell["projected_memory_gib"] <= 1.5
                and cell["control_maintenance_ratio"] <= 0.10)
            cell["actionable_ok"] = bool(
                cell["adam_var_ratio"] <= 0.70 and cell["median_cosine_gain"] >= 0.07
                and cell["projected_memory_gib"] <= 1.5
                and cell["control_maintenance_ratio"] <= 0.15)
            cells[f"{scene}@{stage}"] = cell
        per_K[str(K)] = {
            "cells": cells,
            "all_cells_strong": all(c["strong_ok"] for c in cells.values()),
            "all_cells_actionable": all(c["actionable_ok"] for c in cells.values()),
            "n_strong_cells": sum(c["strong_ok"] for c in cells.values()),
            "n_actionable_cells": sum(c["actionable_ok"] for c in cells.values()),
        }

    passing_strong = [K for K in C.K_LIST if per_K[str(K)]["all_cells_strong"]]
    passing_actionable = [K for K in C.K_LIST if per_K[str(K)]["all_cells_actionable"]]
    if passing_strong:
        result, Kstar = "MG0_STRONG_VARIANCE_SIGNAL", min(passing_strong)
    elif passing_actionable:
        result, Kstar = "MG0_ACTIONABLE_VARIANCE_SIGNAL", min(passing_actionable)
    else:
        result, Kstar = "MG0_NO_USEFUL_VARIANCE_SIGNAL", None

    out = {"campaign": "MG0_OneView_GS_HotCV", "stage": "gate",
           "verdict": result, "K_star": Kstar,
           "run_one_step": bool(Kstar is not None),
           "criteria": {
               "strong": {"adam_var_ratio": "<= 0.50", "median_cosine_gain": ">= +0.12",
                          "positive_dot": "not decreased",
                          "projected_memory_gib": "<= 1.5",
                          "control_maintenance_ratio": "<= 0.10"},
               "actionable": {"adam_var_ratio": "<= 0.70",
                              "median_cosine_gain": ">= +0.07",
                              "projected_memory_gib": "<= 1.5",
                              "control_maintenance_ratio": "<= 0.15"},
               "cells": [f"{s}@{c}" for s, c in CELLS]},
           "per_K": per_K,
           "passing_strong": passing_strong, "passing_actionable": passing_actionable,
           "source_files": {"MG0_VARIANCE_ROOM.json": True, "MG0_VARIANCE_BICYCLE.json": True,
                            "MG0_COST.json": True, "MG0_MEMORY.json": True},
           "script_sha256": {"mg0_gate.py": C.sha256_file(os.path.abspath(__file__))}}
    C.json_dump(out, os.path.join(C.ARTIFACTS, "MG0_GATE.json"))
    print(f"[mg0_gate] {result} K_star={Kstar}", flush=True)


if __name__ == "__main__":
    main()
