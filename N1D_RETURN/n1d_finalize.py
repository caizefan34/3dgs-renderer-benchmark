#!/usr/bin/env python3
"""N1D finalizer: merged timing JSON, raw-row parquet, execution sanity
cross-check vs N1 (A1-1.00 vs N1 A1 replicas — execution verification only,
NOT a scientific verdict), contamination record, and MANIFEST_N1D.json.

No frontier computation, no verdict: the frozen n1d_analyze.py belongs to
the downstream analyst (DSH_TASK_N1D step 8 + Forbidden list).
"""
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

N1D_ROOT = "/root/epic05_n1d"
N1D_ARTIFACTS = os.path.join(N1D_ROOT, "artifacts")
SCENES = ("room", "bicycle")


def load(p):
    with open(p) as f:
        return json.load(f)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def csv_read(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.reader(f))


# ------------------------------------------------------------- timing merge
def merge_timing():
    out = {"campaign": "N1D_ReBudgetGS_Mechanism_Disambiguation",
           "protocol": "anchor 1 per (scene, checkpoint, action); 10 warm-ups "
                       "+ 30 timed repeats from the restored state (restore "
                       "untimed); CUDA events for phases + host wall total; "
                       "no profiler; GPU otherwise idle",
           "scenes": {s: load(f"{N1D_ARTIFACTS}/N1D_ACTION_TIMING_{s}.json")
                      for s in SCENES}}
    C.json_dump(out, f"{N1D_ARTIFACTS}/N1D_ACTION_TIMING.json")


# ------------------------------------------------------------- parquet
def write_parquet():
    cols = {
        "scene": pa.string(), "checkpoint": pa.int32(), "anchor_id": pa.int32(),
        "iteration": pa.int32(), "v1": pa.int32(),
        "action": pa.string(), "alpha": pa.float64(), "partner": pa.int32(),
        "heldout_fused_before": pa.float64(), "heldout_fused_after": pa.float64(),
        "delta_fused": pa.float64(), "delta_loss": pa.float64(),
        "heldout_l1_before": pa.float64(), "heldout_l1_after": pa.float64(),
        "delta_l1": pa.float64(),
        "heldout_psnr_before": pa.float64(), "heldout_psnr_after": pa.float64(),
        "loss_v1": pa.float64(), "loss_v2": pa.float64(),
        "wall_ms": pa.float64(), "total_ms": pa.float64(),
        "upd_norm_global": pa.float64(), "upd_norm_xyz": pa.float64(),
    }
    data = {k: [] for k in cols}
    n = 0
    for s in SCENES:
        d = load(f"{N1D_ARTIFACTS}/N1D_ACTION_RESULTS_{s}.json")
        for r in d["rows"]:
            for k in cols:
                if k == "delta_loss":
                    data[k].append(r["delta_fused"])
                elif k == "total_ms":
                    data[k].append(r["wall_ms"])
                else:
                    data[k].append(r.get(k))
            n += 1
    tbl = pa.table({k: pa.array(data[k], type=t) for k, t in cols.items()})
    pq.write_table(tbl, f"{N1D_ARTIFACTS}/N1D_ACTION_RESULTS.parquet")
    print(f"[n1d] N1D_ACTION_RESULTS.parquet rows={n}")

    # required-actions completeness check (analyzer contract)
    need = {"A0", "A1-1.00", "A1-0.75", "A1-0.50", "A1-0.25",
            "A2-r1", "A2-r2", "A2-r3"}
    by_anchor = {}
    for s in SCENES:
        d = load(f"{N1D_ARTIFACTS}/N1D_ACTION_RESULTS_{s}.json")
        for r in d["rows"]:
            by_anchor.setdefault((s, r["checkpoint"], r["anchor_id"]),
                                 set()).add(r["action"])
    bad = [k for k, acts in by_anchor.items() if not need <= acts]
    n_anchors = len(by_anchor)
    print(f"[n1d] completeness: {n_anchors} anchors, "
          f"all-8-actions={n_anchors - len(bad)}/{n_anchors}")
    if bad:
        print("[n1d] INCOMPLETE ANCHORS:", bad[:10])
        sys.exit(1)
    # action label normalization check (frozen analyzer canonical_action)
    aliases = {"a0", "a1-1.00", "a1-0.75", "a1-0.50", "a1-0.25",
               "a2-r1", "a2-r2", "a2-r3"}
    labels = {r["action"].strip().lower()
              for s in SCENES
              for r in load(f"{N1D_ARTIFACTS}/N1D_ACTION_RESULTS_{s}.json")["rows"]}
    assert labels <= aliases, f"non-canonical action labels: {labels}"
    return n


# ------------------------------------------------------------- sanity cross-check
def n1_crosscheck():
    """Execution verification only: N1D A1-1.00 deltas must agree with N1's
    A1 replica measurements to within execution noise (no verdict drawn)."""
    out = {}
    for s in SCENES:
        n1 = load(f"{C.ARTIFACTS}/N1_ACTION_RESULTS_{s}.json")["rows"]
        n1_map = {}
        for r in n1:
            if r["action"].startswith("A1_r"):
                n1_map.setdefault((r["checkpoint"], r["anchor_id"]),
                                  []).append(r["delta_fused"])
        d = load(f"{N1D_ARTIFACTS}/N1D_ACTION_RESULTS_{s}.json")["rows"]
        diffs, spreads = [], []
        for r in d:
            if r["action"] == "A1-1.00":
                rep = n1_map[(r["checkpoint"], r["anchor_id"])]
                med = float(np.median(rep))
                diffs.append(abs(r["delta_fused"] - med))
                spreads.append(max(rep) - min(rep))
        out[s] = {
            "n_compared": len(diffs),
            "max_abs_dev_vs_n1_replica_median": float(np.max(diffs)),
            "median_abs_dev": float(np.median(diffs)),
            "max_n1_replica_spread": float(np.max(spreads)),
            "note": "execution-noise agreement check only (N1 replicas were "
                    "run from the same restored states); not a verdict",
        }
        print(f"[n1d] crosscheck {s}: max|dev|={out[s]['max_abs_dev_vs_n1_replica_median']:.3e} "
              f"(N1 replica spread max {out[s]['max_n1_replica_spread']:.3e})")
    C.json_dump(out, f"{N1D_ARTIFACTS}/N1D_EXECUTION_SANITY_VS_N1.json")
    return out


# ------------------------------------------------------------- contamination
def contamination():
    end_gpu = sh("nvidia-smi --query-gpu=timestamp,index,memory.used,"
                 "utilization.gpu --format=csv,noheader").strip()
    with open(f"{N1D_ROOT}/cont_campaign_end.csv", "w") as f:
        f.write(end_gpu + "\n")
    out = {
        "campaign": "N1D_ReBudgetGS_Mechanism_Disambiguation",
        "host": "EPIC-05 (8x A100-SXM4-80GB)",
        "principle": "N1D GPU work ran on GPU0 (room) and GPU1 (bicycle); "
                     "GPUs 2-7 untouched; /root/epic05_p12 (canonical) and "
                     "/root/epic05_n1 (N1 artifacts) were read-only for N1D.",
        "gpu_snapshot_campaign_start": csv_read(f"{N1D_ROOT}/cont_gpu_start.csv"),
        "gpu_snapshot_campaign_end": end_gpu.splitlines(),
        "gpu_snapshots_timing": {
            s: {"start": csv_read(f"{N1D_ROOT}/cont_timing_{s}_start.csv"),
                "end": csv_read(f"{N1D_ROOT}/cont_timing_{s}_end.csv")}
            for s in SCENES},
        "uptime_end": sh("uptime").strip(),
        "concurrent_processes_at_end": sh("ps aux | grep -E 'python|trainer' | "
                                          "grep -v grep").strip().splitlines(),
        "n1d_runs": {
            "pass_actions": "GPU0=room / GPU1=bicycle in parallel; 8 actions "
                            "x 192 anchors per scene from restored N1 states",
            "timing": "per scene after that scene's actions; 10 warm-ups + "
                      "30 repeats; GPU otherwise idle; no profiler",
            "training": "NONE — N1D launches no training; it reuses the N1 "
                        "snapshot checkpoints read-only",
        },
        "n1_artifacts_modified": False,
        "note": "StateMaster restore + held-out render determinism were "
                "verified in N1 Stage 0 on the same checkpoints; N1D adds "
                "an LR-invariant assertion after every action.",
    }
    C.json_dump(out, f"{N1D_ARTIFACTS}/N1D_CONTAMINATION.json")


# ------------------------------------------------------------- manifest
def manifest():
    REQ = [
        ("N1D_SOURCE_IDENTITY.json", "N1 identity re-verification + N1D seed/LR-scaling semantics."),
        ("N1D_PARTNER_LIST.json", "Three deterministic frozen random partners per anchor (new seed 20260929), written before any N1D action."),
        ("N1D_ACTION_RESULTS.parquet", "Raw counterfactual rows: A0, A1-1.00/0.75/0.50/0.25, A2-r1/r2/r3 x 192 anchors x 2 scenes."),
        ("N1D_ACTION_TIMING.json", "Representative timings: 10 warm-ups + 30 repeats, CUDA events + host wall."),
        ("N1D_CONTAMINATION.json", "GPU/uptime/process snapshots; contamination audit."),
        ("MANIFEST_N1D.json", "This file."),
    ]
    SUP = [
        ("N1D_ACTION_RESULTS_room.json", "Raw rows, room (JSON)."),
        ("N1D_ACTION_RESULTS_bicycle.json", "Raw rows, bicycle (JSON)."),
        ("N1D_ACTION_TIMING_room.json", "Per-scene timing, room."),
        ("N1D_ACTION_TIMING_bicycle.json", "Per-scene timing, bicycle."),
        ("N1D_EXECUTION_SANITY_VS_N1.json", "A1-1.00 vs N1 A1 replica agreement (execution verification, not a verdict)."),
        ("n1d_analyze.py", "FROZEN analyzer from the N1D bundle (downstream analyst runs it; N1D does not)."),
        ("N1D_PREREG.md", "Frozen N1D preregistration from the bundle."),
        ("N1D_SPEC_EXECUTION.json", "N1D execution notes (seed, draw rule, LR semantics, analyzer contract)."),
    ]
    CSV = [
        ("cont_gpu_start.csv", "GPU snapshot at N1D campaign start."),
        ("cont_campaign_end.csv", "GPU snapshot at N1D campaign end."),
        ("cont_timing_room_start.csv", "GPU snapshot at room timing start."),
        ("cont_timing_room_end.csv", "GPU snapshot at room timing end."),
        ("cont_timing_bicycle_start.csv", "GPU snapshot at bicycle timing start."),
        ("cont_timing_bicycle_end.csv", "GPU snapshot at bicycle timing end."),
    ]
    SCRIPTS = [
        ("n1d_freeze_partners.py", "Identity verification + partner freeze (before any action)."),
        ("n1d_pass_actions.py", "Counterfactual action pass."),
        ("n1d_timing.py", "Timing pass."),
        ("n1d_finalize.py", "Parquet + contamination + manifest."),
    ]

    def entry(base, name, desc):
        p = os.path.join(base, name)
        if name == "MANIFEST_N1D.json":
            return {"file": name, "sha256": "self-referential", "description": desc}
        if not os.path.exists(p):
            return {"file": name, "missing": True, "description": desc}
        return {"file": name, "bytes": os.path.getsize(p),
                "sha256": sha256_file(p), "description": desc}

    out = {
        "campaign": "N1D_ReBudgetGS_Mechanism_Disambiguation",
        "execution_only": True,
        "verdict_computed": False,
        "analyzer": "n1d_analyze.py (frozen in the bundle; run by the "
                    "downstream analyst on N1D_ACTION_RESULTS.parquet; "
                    "suggested --noise from N1 sigma_replica: "
                    "room 8.6e-10, bicycle 1.4e-9 — analyst's choice)",
        "required_artifacts": [entry(N1D_ARTIFACTS, n, d) for n, d in REQ],
        "supporting_evidence": [entry(N1D_ARTIFACTS, n, d) for n, d in SUP],
        "contamination_csvs": [entry(N1D_ROOT, n, d) for n, d in CSV],
        "scripts": [entry(N1D_ROOT, n, d) for n, d in SCRIPTS],
        "reused_n1_artifacts_sha256": {
            "N1_PROBE_LIST.json": sha256_file(f"{C.ARTIFACTS}/N1_PROBE_LIST.json"),
            "note": "anchors + held-out sets reused exactly; N1 artifacts read-only",
        },
    }
    missing = [e["file"] for e in (out["required_artifacts"]
                                   + out["supporting_evidence"]
                                   + out["contamination_csvs"]
                                   + out["scripts"]) if e.get("missing")]
    C.json_dump(out, f"{N1D_ARTIFACTS}/MANIFEST_N1D.json")
    if missing:
        print("MISSING:", missing)
        sys.exit(1)
    print("[n1d] MANIFEST COMPLETE: all files present")


def main():
    # execution-notes spec (small, written by the executor, frozen facts)
    C.json_dump({
        "n1d_seed": 20260929,
        "partner_draw": "per scene rng=random.Random(N1D_SEED) over N1 frozen "
                        "anchor order; rng.sample(pool_ids, 3) without "
                        "replacement; A2-rk uses partners[k-1]",
        "lr_scaling": "absolute set of all 5 param-group lrs to master*alpha "
                      "before the action and back to master after; "
                      "update_learning_rate(C+1) re-established per action; "
                      "Adam moments/beta/eps/step untouched; asserted "
                      "after every action",
        "analyzer_contract": "row fields scene/checkpoint/anchor_id/action/"
                             "delta_fused/wall_ms with canonical labels "
                             "A0, A1-1.00, A1-0.75, A1-0.50, A1-0.25, "
                             "A2-r1, A2-r2, A2-r3; one row per action per "
                             "anchor; delta_loss and total_ms alias columns "
                             "included for robustness",
    }, f"{N1D_ARTIFACTS}/N1D_SPEC_EXECUTION.json")
    merge_timing()
    write_parquet()
    n1_crosscheck()
    contamination()
    manifest()
    print("[n1d] FINALIZE COMPLETE — raw artifacts only, STOP per DSH_TASK_N1D")


if __name__ == "__main__":
    main()
