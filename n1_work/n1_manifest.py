#!/usr/bin/env python3
"""Build MANIFEST_N1.json: every N1 artifact with sha256, bytes, description."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1_core as C

REQUIRED = [
    ("N1_SOURCE_IDENTITY.json", "Frozen sources + snapshot-run provenance (merged scenes)."),
    ("N1_CHECKPOINT_IDENTITY.json", "Checkpoint hashes, restore/determinism audit results, Adam-step semantics (merged scenes)."),
    ("N1_PROBE_LIST.json", "Frozen deterministic probe list (anchors, pools, held-out, unions) written before any gradient."),
    ("N1_ACTION_RESULTS.parquet", "4,608 counterfactual action rows (A0, A1 x3 replicas, A2_random, A2_pool x7)."),
    ("N1_ACTION_TIMING.json", "Stage-3 timing: 10 warm-ups + 30 repeats per (scene, ckpt, action), CUDA-event phases."),
    ("N1_GRADIENT_CONFLICT.parquet", "2,688 true-gradient pair metrics (per-group/global cosines, norm ratios, visibility Jaccard)."),
    ("N1_SKETCH_PREDICTIVITY.json", "Stale vs fresh sketch Spearman against true-gradient disagreement, per scene/checkpoint/tier."),
    ("N1_ACTION_HETEROGENEITY.json", "H1/H2/H3 gates, sigma_replica/sigma_onestep, per-checkpoint aggregates, A2-best tier histogram."),
    ("N1_BUDGET_FRONTIER.json", "Exact MCKP frontier at rho .60/.75/.90, primary + oracle-partner sets, frozen baselines, strong gate."),
    ("N1_CROSS_SCENE_POLICY.json", "Stage-6 gate record: NOT_RUN_GATE_FAILED (strong gate failed on bicycle)."),
    ("N1_CONTAMINATION.json", "GPU/uptime/process snapshots around the campaign; contamination audit."),
    ("N1_VERDICT.json", "Prereg section-11 rule chain with evidence: N1_SMALL_BUT_ACTIONABLE."),
    ("N1_REPORT.md", "Final N1 report."),
    ("MANIFEST_N1.json", "This file."),
]
SUPPORTING = [
    ("N1_SPEC_OPERATIONALIZATION.json", "Frozen operationalization (precedes all measurements)."),
    ("N1_AUDIT_RAW_room.json", "Stage-0 raw audit, room."),
    ("N1_AUDIT_RAW_bicycle.json", "Stage-0 raw audit, bicycle."),
    ("N1_SOURCE_IDENTITY_room.json", "Per-scene source identity, room."),
    ("N1_SOURCE_IDENTITY_bicycle.json", "Per-scene source identity, bicycle."),
    ("N1_CHECKPOINT_IDENTITY_room.json", "Per-scene checkpoint identity, room."),
    ("N1_CHECKPOINT_IDENTITY_bicycle.json", "Per-scene checkpoint identity, bicycle."),
    ("N1_ACTION_RESULTS_room.json", "Raw action rows, room (JSON backup of parquet subset)."),
    ("N1_ACTION_RESULTS_bicycle.json", "Raw action rows, bicycle."),
    ("N1_ACTION_TIMING_room.json", "Per-scene timing, room."),
    ("N1_ACTION_TIMING_bicycle.json", "Per-scene timing, bicycle."),
    ("N1_GRADIENT_CONFLICT_room.json", "Raw conflict rows, room."),
    ("N1_GRADIENT_CONFLICT_bicycle.json", "Raw conflict rows, bicycle."),
    ("N1_SKETCHES_room.npz", "1,014 sketches (1024-dim) at 10/15/20/25K, room."),
    ("N1_SKETCHES_bicycle.npz", "712 sketches (1024-dim) at 10/15/20/25K, bicycle."),
    ("N1_VIEW_LOSSES_room.json", "Per-view fused losses at each snapshot (EMA chain), room."),
    ("N1_VIEW_LOSSES_bicycle.json", "Per-view fused losses at each snapshot (EMA chain), bicycle."),
]
ROOT_FILES = [
    ("publication_trainer_p12_n1snap.py", "Snapshot trainer (save-only edit of frozen P12 trainer)."),
    ("n1_core.py", "Probe core: bootstrap_c0, fused loss, render, StateMaster, Sketcher."),
    ("n1_stage1_probes.py", "Stage-1 probe-list builder."),
    ("n1_stage0_audit.py", "Stage-0 audit."),
    ("n1_pass_grads.py", "Stage-2a/2b pass (sketches + conflict ground truth)."),
    ("n1_pass_actions.py", "Stage-2c counterfactual action pass."),
    ("n1_stage3_timing.py", "Stage-3 timing pass."),
    ("n1_stage455_analysis.py", "Stages 4+5 analysis (H1/H2/H3, sketches, frontier)."),
    ("n1_finalize.py", "Merges, parquet serialization, verdict."),
    ("n1_manifest.py", "Manifest builder (this run)."),
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    def entry(base, name, desc):
        p = os.path.join(base, name)
        if name == "MANIFEST_N1.json":
            return {"file": name, "sha256": "self-referential",
                    "description": desc}
        if not os.path.exists(p):
            return {"file": name, "missing": True, "description": desc}
        return {"file": name, "bytes": os.path.getsize(p),
                "sha256": sha256_file(p), "description": desc}

    out = {
        "campaign": "N1_ReBudgetGS_mechanism_measurement",
        "verdict": "N1_SMALL_BUT_ACTIONABLE",
        "required_artifacts": [entry(C.ARTIFACTS, n, d) for n, d in REQUIRED],
        "supporting_evidence": [entry(C.ARTIFACTS, n, d) for n, d in SUPPORTING],
        "scripts_and_specs": [entry(C.N1_ROOT, n, d) for n, d in ROOT_FILES],
        "contamination_csvs": [entry(C.N1_ROOT, n, d) for n, d in [
            ("cont_gpu_start.csv", "GPU snapshot at campaign start."),
            ("cont_stage23_end.csv", "GPU snapshot at Stage 2+3 pipeline end."),
            ("cont_campaign_end.csv", "GPU snapshot at campaign end."),
            ("cont_timing_room_start.csv", "GPU snapshot at room timing start."),
            ("cont_timing_room_end.csv", "GPU snapshot at room timing end."),
            ("cont_timing_bicycle_start.csv", "GPU snapshot at bicycle timing start."),
            ("cont_timing_bicycle_end.csv", "GPU snapshot at bicycle timing end."),
        ]],
    }
    missing = [e["file"] for e in (out["required_artifacts"] + out["supporting_evidence"]
                                   + out["scripts_and_specs"])
               if e.get("missing")]
    C.json_dump(out, os.path.join(C.ARTIFACTS, "MANIFEST_N1.json"))
    if missing:
        print("MISSING:", missing)
        sys.exit(1)
    print("MANIFEST COMPLETE: all files present")


if __name__ == "__main__":
    main()
