#!/usr/bin/env python3
"""OPTLAWS0 / OC0 step 7 — mechanical application of the frozen discovery-label
rule.

Usage: oc0_gate.py

Consumes OC0_ANALYSIS.json (prereg_criteria_check) and applies the OC0_PREREG
rules verbatim:

  OC0_STRONG_OCCLUSION_CONFLICT_STRUCTURE : all of
     (1) conflict nontrivial in every cell (median hot negative-pair fraction
         >= 0.10);
     (2) at least one T/depth-role feature with |Spearman| >= 0.35 in all four
         cells;
     (3) every cross-scene/temporal predictor ROC AUC >= 0.75;
     (4) top-decile enrichment >= 2.0x in every test cell.

  OC0_MODERATE_STRUCTURE : nontrivial conflict AND every AUC >= 0.65 AND
     top-decile enrichment >= 1.5x.

  OC0_NO_PREDICTABLE_OCCLUSION_STRUCTURE : otherwise.

The label is a frozen mechanical consequence of the preregistered thresholds;
per DSH_TASK_OPTLAWS0.md no scientific verdict is issued and raw artifacts are
returned for independent analysis.

Emits
-----
  OC0_GATE.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")))
import mg0_core as C  # noqa: E402
import oc0_common as K  # noqa: E402


def main():
    t0 = time.time()
    ana = json.load(open(K.artifact("OC0_ANALYSIS.json")))
    crit = ana["prereg_criteria_check"]

    c1 = crit["criterion_1_nontrivial_conflict"]["all_cells_satisfied"]
    c2 = crit["criterion_2_t_depth_role_spearman"]["all_cells_satisfied"]
    c3 = crit["criterion_3_auc"]
    c4 = crit["criterion_4_enrichment"]

    strong = bool(c1 and c2 and c3["strong_satisfied"] and c4["strong_satisfied"])
    moderate = bool(
        c1 and not strong
        and c3["moderate_satisfied"] and c4["moderate_satisfied"])
    if strong:
        label = "OC0_STRONG_OCCLUSION_CONFLICT_STRUCTURE"
    elif moderate:
        label = "OC0_MODERATE_STRUCTURE"
    else:
        label = "OC0_NO_PREDICTABLE_OCCLUSION_STRUCTURE"

    out = {
        "campaign": "OPTLAWS0", "stage": "OC0_gate",
        "prereg": "OC0_PREREG.md",
        "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_analysis_sha256": C.sha256_file(K.artifact("OC0_ANALYSIS.json")),
        "rules": {
            "OC0_STRONG_OCCLUSION_CONFLICT_STRUCTURE": [
                "median hot negative-pair fraction >= 0.10 in all cells",
                "some T/depth-role feature |Spearman| >= 0.35 in all four cells",
                "all cross-scene/temporal ROC AUC >= 0.75",
                "top-decile enrichment >= 2.0x in every test cell"],
            "OC0_MODERATE_STRUCTURE": [
                "nontrivial conflict",
                "all cross-scene/temporal AUC >= 0.65",
                "top-decile enrichment >= 1.5x"],
            "OC0_NO_PREDICTABLE_OCCLUSION_STRUCTURE": ["otherwise"],
        },
        "criterion_evaluation": {
            "c1_nontrivial_conflict": c1,
            "c2_t_depth_role_spearman": c2,
            "c3_auc_strong": c3["strong_satisfied"],
            "c3_auc_moderate": c3["moderate_satisfied"],
            "c4_enrichment_strong": c4["strong_satisfied"],
            "c4_enrichment_moderate": c4["moderate_satisfied"],
        },
        "discovery_label": label,
        "note": ("mechanical application of the frozen preregistered rule; "
                 "NOT a scientific verdict. Per DSH_TASK_OPTLAWS0.md raw "
                 "artifacts are returned for independent analysis."),
        "timing_s": time.time() - t0,
        "script_sha256": {os.path.basename(__file__):
                          C.sha256_file(os.path.abspath(__file__))},
    }
    C.json_dump(out, K.artifact("OC0_GATE.json"))
    print(f"[oc0_gate] COMPLETE label={label}", flush=True)


if __name__ == "__main__":
    main()