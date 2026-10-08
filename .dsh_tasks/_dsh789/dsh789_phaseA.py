#!/usr/bin/env python3
"""DSH-7/8/9 Phase-A aggregator.

Reads phase_a_records.jsonl from all three scene trace directories and produces
the three Phase-A structural trace artifacts plus the DSH-8 P1 headroom gate
evaluation. Behavior-inert analysis only; reads the JSONL the tracer wrote.

No training-semantics change, no threshold shopping, no scene-specific rescue.
Authoritative GPU is EPIC-05 A100 (the data was captured there).
"""
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path("/root/dsh_20261001/runs/DSH_PHASEA")
SCENES = ["room", "garden", "bicycle"]
K = 32
TILE = 16
BLOCK = TILE * TILE

# Mature iterations used for headline structural-gate decisions (spec lists
# 500/2K/5K/10K/14.5K/20K/25K; the headroom gate must hold at maturity).
MATURE_ITERS = [10000, 14500, 20000, 25000]


def load_scene(scene):
    recs = []
    p = RUNS / scene / "trace" / "phase_a_records.jsonl"
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["iteration"])
    return recs


def main():
    all_scenes = {s: load_scene(s) for s in SCENES}
    for s in SCENES:
        print(f"loaded {s}: {len(all_scenes[s])} records, "
              f"iters={[r['iteration'] for r in all_scenes[s]]}")

    # ============================ DSH-7 BARS0 Phase A ============================
    bars = {
        "p0": "DSH-7 BARS0 Bucket-Adaptive Replay Scheduling",
        "phase": "A_structural_trace",
        "K": K,
        "scenes": {},
        "note": ("Per-tile structural distributions across representative "
                 "iterations. Feeds Phase B static-K microbench and Phase C "
                 "adaptive oracle; no timing claim made here."),
    }
    for s in SCENES:
        recs = all_scenes[s]
        if not recs:
            continue
        splat = [r["dsh7_bars0"]["splat_list_len"] for r in recs]
        maxc = [r["dsh7_bars0"]["max_contrib"] for r in recs]
        nbkt = [r["dsh7_bars0"]["n_k32_buckets"] for r in recs]
        occ = [r["dsh7_bars0"]["partial_bucket_occupancy"] for r in recs]
        cv = []
        for r in recs:
            m = r["dsh7_bars0"]["splat_list_len"]["mean"]
            p10 = r["dsh7_bars0"]["splat_list_len"]["p10"]
            p90 = r["dsh7_bars0"]["splat_list_len"]["p90"]
            cv.append((p90 - p10) / max(m, 1.0))
        bars["scenes"][s] = {
            "n_records": len(recs),
            "iterations": [r["iteration"] for r in recs],
            "splat_list_len": splat,
            "max_contrib": maxc,
            "n_k32_buckets": nbkt,
            "partial_bucket_occupancy": occ,
            "heterogeneity_p90_minus_p10_over_mean": {
                "min": float(np.min(cv)), "max": float(np.max(cv)),
                "mean": float(np.mean(cv)),
            },
            "final_record": recs[-1],
        }
    (RUNS.parent / "artifacts" / "DSH7_BARS0").mkdir(parents=True, exist_ok=True)
    bars_path = RUNS.parent / "artifacts" / "DSH7_BARS0" / "BARS0_TILE_TRACE.json"
    bars_path.write_text(json.dumps(bars, indent=2), encoding="utf-8")
    print(f"wrote {bars_path}")

    # ============================ DSH-8 APBC0 Phase A ============================
    reach = {
        "p0": "DSH-8 APBC0 Exact Active-Pixel Bucket Compaction",
        "phase": "A_structural_headroom_trace",
        "K": K,
        "predicate": "pixel p reachable by bucket b iff n_contrib[p] > 32*b "
                     "(verified vs kernels_backward.cuh blend_backward_cu: "
                     "bucket survives iff tile_bucket_idx*32 < tile_max_n_processed; "
                     "lane skips iff tile_primitive_idx >= last_contributor)",
        "scenes": {},
        "gate_P1": {
            "description": ">=25% of baseline bucket-pixel scan visits are "
                           "structurally unreachable, cost-weighted, in EVERY scene.",
            "threshold": 0.25,
            "per_scene_mature": {},
            "every_scene_pass": False,
            "min_scene_mature": None,
        },
    }
    per_scene_mature_frac = {}
    for s in SCENES:
        recs = all_scenes[s]
        mature = [r for r in recs if r["iteration"] in MATURE_ITERS]
        use = mature if mature else recs
        fracs = {}
        for r in recs:
            d8 = r["dsh8_apbc0"]
            fracs[r["iteration"]] = d8["unreachable_fraction_cost_weighted"]
        mature_fracs = {r["iteration"]: r["dsh8_apbc0"]
                        ["unreachable_fraction_cost_weighted"]
                        for r in use}
        per_scene_mature_frac[s] = mature_fracs
        reach["scenes"][s] = {
            "n_records": len(recs),
            "iterations": [r["iteration"] for r in recs],
            "unreachable_fraction_by_iter": fracs,
            "mature_unreachable_fraction": mature_fracs,
            "mature_records": [r["dsh8_apbc0"] for r in use],
            "final_record": recs[-1]["dsh8_apbc0"] if recs else None,
        }
    scene_max = {}
    for s, mf in per_scene_mature_frac.items():
        scene_max[s] = max(mf.values()) if mf else 0.0
    every_pass = bool(scene_max) and all(v >= 0.25 for v in scene_max.values())
    reach["gate_P1"]["per_scene_mature_max"] = scene_max
    reach["gate_P1"]["every_scene_pass"] = every_pass
    reach["gate_P1"]["min_scene_mature"] = min(scene_max.values()) if scene_max else None
    reach["gate_P1"]["decision"] = (
        "APBC0_P1_PASS" if every_pass else "APBC0_NO_STRUCTURAL_HEADROOM")
    (RUNS.parent / "artifacts" / "DSH8_APBC0").mkdir(parents=True, exist_ok=True)
    reach_path = RUNS.parent / "artifacts" / "DSH8_APBC0" / "APBC0_REACHABILITY_TRACE.json"
    reach_path.write_text(json.dumps(reach, indent=2), encoding="utf-8")
    print(f"wrote {reach_path}")
    print(f"DSH-8 P1 per-scene mature max unreachable frac: {scene_max}")
    print(f"DSH-8 P1 decision: {reach['gate_P1']['decision']}")

    # ============================ DSH-9 CVL0 Phase A ============================
    cvl = {
        "p0": "DSH-9 CVL0 Co-Visibility Locality Layout Audit",
        "phase": "A_access_trace",
        "coaccess_rule": "adjacent primitive ids within the same sorted tile "
                         "list (spec-permitted); no full NxN graph stored.",
        "scenes": {},
    }
    for s in SCENES:
        recs = all_scenes[s]
        if not recs:
            continue
        edges = [r["dsh9_cvl0"]["n_coaccess_edges_unique"] for r in recs]
        vis = [r["dsh9_cvl0"]["primitive_visible_frac_gt0"] for r in recs]
        mtp = [r["dsh9_cvl0"]["mean_tiles_per_visible"] for r in recs]
        cvl["scenes"][s] = {
            "n_records": len(recs),
            "iterations": [r["iteration"] for r in recs],
            "n_coaccess_edges_unique": edges,
            "primitive_visible_frac_gt0": vis,
            "mean_tiles_per_visible": mtp,
            "coaccess_edge_weight_final": recs[-1]["dsh9_cvl0"]["coaccess_edge_weight"],
            "visibility_freq_final": recs[-1]["dsh9_cvl0"]["visibility_freq"],
            "final_record": recs[-1]["dsh9_cvl0"],
        }
    (RUNS.parent / "artifacts" / "DSH9_CVL0").mkdir(parents=True, exist_ok=True)
    cvl_path = RUNS.parent / "artifacts" / "DSH9_CVL0" / "CVL0_ACCESS_TRACE.json"
    cvl_path.write_text(json.dumps(cvl, indent=2), encoding="utf-8")
    print(f"wrote {cvl_path}")

    print("DONE")


if __name__ == "__main__":
    main()
