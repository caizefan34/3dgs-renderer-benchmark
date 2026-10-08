#!/usr/bin/env python3
"""OPTLAWS0 / MG0-S gate — mechanical evaluation of the frozen discovery labels.

Reads the four MG0-S cells (room/bicycle at 20K/25K) produced by
mg0s_variance.py + mg0s_lowrank.py and emits the descriptive labels defined in
MG0S_PREREG (Part A "Strong/Moderate L1", Part B "Strong/Moderate L2", and the
combined OPT_STRUCTURE_STRONG). No threshold is chosen or tuned here: every
number is a literal prereg constant.

This emits descriptive discovery labels only. It does not issue a scientific
verdict; interpretation stays with the independent analysis of the raw
artifacts.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")))
import mg0_core as C  # noqa: E402

CELLS = (("room", 20000), ("room", 25000), ("bicycle", 20000), ("bicycle", 25000))
TOP2, TOP5, TOP10, TOP20 = "2%", "5%", "10%", "20%"


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    conc, low = {}, {}
    for scene, stage in CELLS:
        conc[(scene, stage)] = _load(os.path.join(
            C.ARTIFACTS, f"MG0S_CONCENTRATION_{scene}_{stage}.json"))
        low[(scene, stage)] = _load(os.path.join(
            C.ARTIFACTS, f"MG0S_LOWRANK_{scene}_{stage}.json"))

    per_cell = {}
    for scene, stage in CELLS:
        c = conc[(scene, stage)]
        l = low[(scene, stage)]
        n_rows = c["N_rows"]
        top2 = c["overall"]["cumulative_top_fraction"][TOP2]["energy_fraction"]
        top5 = c["overall"]["cumulative_top_fraction"][TOP5]["energy_fraction"]
        rows90 = c["overall"]["rows_for_energy"]["90%"]
        frac90 = (rows90 / n_rows) if rows90 else 1.0
        r10 = c["overall"]["cumulative_top_fraction"][TOP10]["energy_fraction"]
        r20 = c["overall"]["cumulative_top_fraction"][TOP20]["energy_fraction"]

        vis = l["modes"]["visible_only"]
        av = l["modes"]["all_views"]
        cell = {
            "N_rows": n_rows, "n_zero_variance_rows": c["n_zero_variance_rows"],
            "top2pct_energy": top2, "top5pct_energy": top5,
            "top10pct_energy": r10, "top20pct_energy": r20,
            "rows_for_90pct": rows90, "rows_for_90pct_fraction": frac90,
            "gini": c["overall"]["gini"],
            "effective_support": c["overall"]["effective_support"],
            "visible_only_median_rank8_explained_hot": vis["hot"].get("median_rank8_explained"),
            "visible_only_median_rank8_explained_uniform": vis["uniform"].get("median_rank8_explained"),
            "visible_only_median_r_eff_hot": vis["hot"].get("median_r_eff"),
            "visible_only_median_r_eff_uniform": vis["uniform"].get("median_r_eff"),
            "visible_only_median_rank16_explained_hot": vis["hot"].get("median_rank16_explained"),
            "all_views_median_rank8_explained_hot": av["hot"].get("median_rank8_explained"),
            "n_insufficient_visible_only": vis["hot"].get("n_insufficient", 0)
            + vis["uniform"].get("n_insufficient", 0),
            "mg0_score_crosscheck_present": c["mg0_score_crosscheck"]["present"],
        }
        cell["L1_strong_cell"] = bool(top2 >= 0.70 and frac90 <= 0.10)
        cell["L1_moderate_cell"] = bool(top5 >= 0.60 and frac90 <= 0.20)
        cell["L2_strong_cell"] = bool(
            (cell["visible_only_median_rank8_explained_hot"] or 0) >= 0.90
            and (cell["visible_only_median_r_eff_hot"] or 1e9) <= 8.0
            and (cell["visible_only_median_rank8_explained_uniform"] or 0) >= 0.80)
        cell["L2_moderate_cell"] = bool(
            (cell["visible_only_median_rank16_explained_hot"] or 0) >= 0.90
            and (cell["visible_only_median_r_eff_hot"] or 1e9) <= 16.0)
        per_cell[f"{scene}@{stage}"] = cell

    cells = list(per_cell.values())
    l1_strong = all(c["L1_strong_cell"] for c in cells)
    l1_moderate = all(c["L1_moderate_cell"] for c in cells)
    l1 = ("L1_STRONG_CONCENTRATION" if l1_strong else
          "L1_MODERATE_CONCENTRATION" if l1_moderate else "L1_DIFFUSE_VARIANCE")

    l2_strong = all(c["L2_strong_cell"] for c in cells)
    l2_moderate = all(c["L2_moderate_cell"] for c in cells)
    l2 = ("L2_STRONG_LOW_RANK" if l2_strong else
          "L2_MODERATE_LOW_RANK" if l2_moderate else "L2_NO_GENERAL_LOW_RANK")

    combined = bool(l1 in ("L1_STRONG_CONCENTRATION", "L1_MODERATE_CONCENTRATION")
                    and l2 in ("L2_STRONG_LOW_RANK", "L2_MODERATE_LOW_RANK"))

    out = {
        "campaign": "OPTLAWS0", "stage": "MG0S_gate",
        "cells": [f"{s}@{st}" for s, st in CELLS],
        "criteria": {
            "L1_strong": "all cells: top-2% rows capture >= 70% AND <= 10% of rows "
                         "capture >= 90% of Adam-coordinate variance",
            "L1_moderate": "all cells: top-5% capture >= 60% AND <= 20% of rows "
                           "capture >= 90%",
            "L2_strong": "all cells: median visible-only rank-8 explained energy "
                         ">= 90% AND median visible-only r_eff <= 8 AND uniform "
                         "median rank-8 >= 80%",
            "L2_moderate": "all cells: median visible-only rank-16 explained >= 90% "
                           "AND median visible-only r_eff <= 16",
            "OPT_STRUCTURE_STRONG": "L1 strong-or-moderate AND L2 strong-or-moderate "
                                    "across room+bicycle at 20K+25K",
        },
        "per_cell": per_cell,
        "labels": {
            "L1": l1,
            "L2": l2,
            "OPT_STRUCTURE_STRONG": combined,
            "L1_strong_all_cells": l1_strong, "L1_moderate_all_cells": l1_moderate,
            "L2_strong_all_cells": l2_strong, "L2_moderate_all_cells": l2_moderate,
        },
        "verdict_policy": "Descriptive discovery labels only. No scientific "
                          "verdict is issued by the execution harness; raw "
                          "artifacts are returned for independent analysis.",
        "source_files": {f"MG0S_CONCENTRATION_{s}_{st}.json": True
                         for s, st in CELLS} |
                        {f"MG0S_LOWRANK_{s}_{st}.json": True for s, st in CELLS},
        "script_sha256": {"mg0s_gate.py": C.sha256_file(os.path.abspath(__file__))},
    }
    C.json_dump(out, os.path.join(C.ARTIFACTS, "MG0S_GATE.json"))
    print(f"[mg0s_gate] {l1} | {l2} | OPT_STRUCTURE_STRONG={combined}", flush=True)


if __name__ == "__main__":
    main()