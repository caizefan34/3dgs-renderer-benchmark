#!/usr/bin/env python3
"""Build the predeclared C0 materiality and historical-impact matrices."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_json


def main() -> None:
    gates = {
        "population_relative": 0.10,
        "e2e_time_relative": 0.05,
        "psnr_db_absolute": 0.2,
        "ssim_absolute": 0.005,
        "densification_count_relative": 0.10,
        "ordering": "material rank/predictor ordering change",
        "gradient_concentration": "material distributional change",
    }
    rows = [
        {"phase": "C42", "finding": "resolution-adaptive SSIM", "sensitivity": "LOW", "classification": "SAFE", "required_action": "none; topology is common-mode and the claim is loss-kernel local"},
        {"phase": "C44", "finding": "separable/frequency-reduced SSIM", "sensitivity": "LOW", "classification": "SAFE", "required_action": "none for kernel equivalence; label training curves CURRENT_PROJECT_SEMANTICS"},
        {"phase": "C45", "finding": "moderate configuration as canonical training baseline", "sensitivity": "VERY HIGH", "classification": "INVALID_UNDER_REFERENCE", "required_action": "do not reuse as canonical baseline; establish a tracked full-reference trainer"},
        {"phase": "C49", "finding": "gradient concentration and oracle filtering", "sensitivity": "MEDIUM", "classification": "NEEDS_RECALIBRATION", "required_action": "recompute concentration/coverage on the patched Room baseline"},
        {"phase": "C50", "finding": "temporal gradient predictability", "sensitivity": "MEDIUM", "classification": "NEEDS_RECALIBRATION", "required_action": "recompute predictor ordering and coverage on patched Room"},
        {"phase": "C51", "finding": "sparse backward plus population-mediated speedup", "sensitivity": "VERY HIGH", "classification": "NEEDS_RERUN", "required_action": "minimal reference baseline vs K50-B1 Room run after C0 factorial"},
        {"phase": "C52", "finding": "gradient-ranked density allocation does not consistently beat uniform", "sensitivity": "VERY HIGH", "classification": "NEEDS_RERUN", "required_action": "repeat only baseline/uniform/ranked Room conditions under reference semantics"},
        {"phase": "C53", "finding": "workload/utility/persistence mechanism", "sensitivity": "LOW-MEDIUM", "classification": "NEEDS_RECALIBRATION", "required_action": "recompute distributional statistics on one patched Room trajectory; mechanism code remains valid"},
    ]
    payload = {
        "status": "FINAL_AFTER_ROOM_FACTORIAL",
        "materiality_gates_predeclared": gates,
        "historical_results_namespace": "CURRENT_PROJECT_SEMANTICS",
        "rows": rows,
        "factorial_gate_results": {
            "split_mean_final_population_effect": -542.5,
            "split_mean_final_psnr_db": -0.028059304851741018,
            "split_mean_final_ssim": -0.00007316240897542459,
            "adam_final_population_relative_current_split": 0.107454,
            "adam_final_population_relative_reference_split": 0.107593,
            "adam_mean_final_psnr_db": -0.058936937957629,
            "adam_mean_final_ssim": 0.0004406754787151246,
            "adam_prune_count_relative_current_split": -0.61373,
            "interaction_final_population": 123,
            "interaction_final_psnr_db": -0.00040585613454169334,
            "dominant_factor": "optimizer-state handling (population and prune trajectory)",
            "material_gates_crossed": ["population >10%", "prune count >10%"],
        },
        "reproducibility_warning": "The archived C51 5K baseline records 42,538 splits, while the audited current source at threshold 0.001 produces 840 in condition A. The scripts are untracked and result metadata omits the threshold, so exact historical source state is not recoverable from git. Historical magnitude claims therefore require revalidation even where this C0 run is below gate.",
        "minimum_revalidation": [
            "Room 5K C0 factorial A/B/C/D on canonical A100 environment",
            "C49/C50/C53 statistics from the reference-semantics Room trajectory",
            "C51 reference baseline vs K50-B1 Room only",
            "C52 reference baseline vs uniform vs gradient-ranked Room only",
        ],
    }
    print(write_json("historical_impact_matrix.json", payload))
    print(write_json("candidate_reclassification.json", {
        "status": "FINAL",
        "rows": [
            {"candidate": "A — function-preserving fission", "survives_reference_parity": False, "research_status": "DROP", "reason": "Graphdeco already replaces a split parent with children; removing the extra retained parent is baseline repair."},
            {"candidate": "B — transactional optimizer", "survives_reference_parity": False, "research_status": "DROP", "reason": "Graphdeco already preserves survivor Adam moments and zero-initializes appended moment rows; implementing this locally is baseline repair."},
        ],
    }))


if __name__ == "__main__":
    main()
