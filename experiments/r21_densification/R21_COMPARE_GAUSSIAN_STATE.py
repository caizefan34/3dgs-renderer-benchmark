#!/usr/bin/env python3
"""Compare captured state by stable lineage IDs, or find first per-step mismatch.
Status: IMPLEMENTED; actual MX trajectories NOT_EXECUTED locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from R21_STATE import align_ids, digest, load_state, tensor_metrics, write_json, topology_digest


def compare_states(a, b, atol=0.0, rtol=0.0):
    if a["iteration"] != b["iteration"] or a["phase"] != b["phase"]:
        raise ValueError("Compare identical iteration and phase")
    ai, bi = align_ids(a["stable_ids"], b["stable_ids"])
    report = {"schema":"r21.comparison.v1", "iteration":a["iteration"], "phase":a["phase"],
              "left_count":len(a["stable_ids"]), "right_count":len(b["stable_ids"]),
              "shared_lineages":len(ai), "atol":atol, "rtol":rtol,
              "left_only_ids":a["stable_ids"][~torch.isin(a["stable_ids"], b["stable_ids"])].tolist(),
              "right_only_ids":b["stable_ids"][~torch.isin(b["stable_ids"], a["stable_ids"])].tolist(),
              "fields":{}}
    def compare_field(name, x, y, rows=True):
        if x is None or y is None:
            report["fields"][name] = {"equal":x is None and y is None, "left_none":x is None, "right_none":y is None}
            return
        if rows:
            x, y = x[ai], y[bi]
        result = tensor_metrics(x, y, atol, rtol)
        if rows and x.shape == y.shape:
            different = (x != y).reshape(len(ai), -1).any(-1) if len(ai) else torch.zeros(0,dtype=torch.bool)
            result["mismatch_ids"] = a["stable_ids"][ai[different]].tolist()
        report["fields"][name] = result
    for category in ("raw", "raw_grad", "stats"):
        for key in a[category]:
            compare_field(f"{category}.{key}", a[category][key], b[category][key])
    for name in ("visibility", "native_visibility", "statistics_radii"):
        if name in a and name in b:
            compare_field(name, a[name], b[name])
    for name in ("absgrad", "signed_means2d"):
        if name in a and name in b:
            x,y = a[name], b[name]
            compare_field(name, None if x is None else x.reshape(len(a["stable_ids"]),2),
                          None if y is None else y.reshape(len(b["stable_ids"]),2))
    for category in ("candidates", "projection_canonical", "stats_before"):
        if category in a and category in b:
            for key in a[category]:
                x, y = a[category][key], b[category][key]
                if isinstance(x, torch.Tensor):
                    compare_field(f"{category}.{key}", x, y)
    for name, group in a["optimizer"]["state"].items():
        other = b["optimizer"]["state"][name]
        for key in sorted(set(group) | set(other)):
            x,y = group.get(key),other.get(key)
            if isinstance(x,torch.Tensor) or isinstance(y,torch.Tensor):
                compare_field(f"optimizer.{name}.{key}", x, y, key != "step")
            else:
                report["fields"][f"optimizer.{name}.{key}"] = {"equal":x == y}
    report["rng_equal"] = digest(a["rng"]) == digest(b["rng"])
    report["camera_equal"] = digest(a["camera"]) == digest(b["camera"])
    report["optimizer_groups_equal"] = digest(a["optimizer"]["groups"]) == digest(b["optimizer"]["groups"])
    report["topology_events_equal"] = topology_digest(a.get("topology_events",[])) == topology_digest(b.get("topology_events",[]))
    report["actual_decisions"] = {}
    for key in sorted(set(a.get("decisions",{})) | set(b.get("decisions",{}))):
        left, right = a.get("decisions",{}).get(key), b.get("decisions",{}).get(key)
        if left is None or right is None:
            report["actual_decisions"][key] = {"equal":False, "reason":"missing decision phase"}
            continue
        li,ri = align_ids(left["stable_ids"], right["stable_ids"])
        mismatch = left["mask"][li] != right["mask"][ri]
        sets_equal = len(li) == len(left["stable_ids"]) == len(right["stable_ids"])
        report["actual_decisions"][key] = {"equal":sets_equal and not bool(mismatch.any()),
            "left_selected":left["stable_ids"][left["mask"]].tolist(),
            "right_selected":right["stable_ids"][right["mask"]].tolist(),
            "shared_mismatch_ids":left["stable_ids"][li[mismatch]].tolist(), "id_sets_equal":sets_equal}
    report["status"] = "COMPARISON_COMPLETE"
    report["interpretation"] = "Exact first differences are distinct from tolerance failure and causal explanation."
    return report


def read_journal(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("schema") != "r21.step.v2" for row in rows):
        raise ValueError("Journal must use r21.step.v2 lineage-event hashes; recapture legacy journals")
    ids = [r["iteration"] for r in rows]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("Journal has duplicate or unordered steps")
    return {r["iteration"]:r for r in rows}


def compare_journals(left, right):
    a, b = read_journal(left), read_journal(right)
    common = sorted(set(a) & set(b))
    fields = ("raw_before", "raw_after", "radii", "projection", "visibility", "absgrad", "raw_grad",
              "stats_before", "stats_after", "candidates", "actual_decisions", "stable_ids_after", "topology_events", "camera")
    first = {key:next((i for i in common if a[i][key] != b[i][key]), None) for key in fields}
    contiguous_from_one = common == list(range(1, max(common, default=0)+1)) and bool(common)
    return {"schema":"r21.first-divergence.v1", "status":"COMPARISON_COMPLETE", "first_exact_difference":first,
            "first_parameter_divergence":min([v for k,v in first.items() if k in ("raw_before","raw_after") and v is not None],default=None),
            "first_projection_divergence":first["projection"], "first_statistics_divergence":first["stats_after"],
            "first_candidate_mask_divergence":first["candidates"],
            "first_topology_divergence":min((v for v in (first["topology_events"],first["stable_ids_after"]) if v is not None),default=None),
            "first_topology_event_divergence":first["topology_events"],
            "first_survivor_set_divergence":first["stable_ids_after"],
            "first_actual_decision_divergence":first["actual_decisions"],
            "common_steps":common, "left_missing":sorted(set(b)-set(a)), "right_missing":sorted(set(a)-set(b)),
            "proves_first_since_initialization":contiguous_from_one,
            "interpretation":"SHA differences detect exact numerical/structural differences; inspect aligned snapshots for magnitude and cause."}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", required=True, type=Path); ap.add_argument("--right", required=True, type=Path)
    ap.add_argument("--output",required=True,type=Path); ap.add_argument("--journals",action="store_true")
    ap.add_argument("--atol",type=float,default=0.0); ap.add_argument("--rtol",type=float,default=0.0)
    args=ap.parse_args()
    result=compare_journals(args.left,args.right) if args.journals else compare_states(load_state(args.left),load_state(args.right),args.atol,args.rtol)
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("fields","common_steps","actual_decisions","left_only_ids","right_only_ids")},indent=2))


if __name__ == "__main__":
    main()
