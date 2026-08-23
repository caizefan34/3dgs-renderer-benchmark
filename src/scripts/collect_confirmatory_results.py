#!/usr/bin/env python3
"""Collect confirmatory matrix runs into a summary JSON for analysis.

Scans a run directory written by run_confirmatory_matrix.py, loads every
per-run harness JSON, and emits a summary document with a \"runs\" mapping
(run_id -> metrics) compatible with src/scripts/bootstrap_analysis.py plus a
pre-registered time-to-target-quality map.

Time-to-target rule (pre-registered, see paper/confirmatory-protocol.md):
for each (scene, seed) pair, target = the ctrl arm's final PSNR; the pd arm's
time-to-target is the wall_s of the first eval point whose PSNR reaches the
target (None if never reached on the 30k horizon).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_ID_RE = re.compile(
    r"(?P<family>[a-z0-9_]+)_(?P<scene_name>[a-z0-9_]+)_(?P<arm>ctrl|pd)_s(?P<seed>\d+)$"
)

FIELDS = [
    "train_ms", "total_wall_s", "psnr", "ssim", "lpips",
    "final_n", "peak_vram_gb", "culling_ratio", "probe_grad_cosine",
    "probe_init_psnr", "device", "torch",
]


def collect(out_dir: Path) -> dict:
    runs: dict[str, dict] = {}
    for json_path in sorted(out_dir.glob("*.json")):
        if json_path.name in ("manifest.json", "summary.json"):
            continue
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scene_records = list(doc.get("scenes", {}).values())
        if not scene_records:
            continue
        record = scene_records[0][0] if isinstance(scene_records[0], list) else scene_records[0]
        if "error" in record:
            continue
        match = RUN_ID_RE.match(json_path.stem)
        if not match:
            continue
        run_id = json_path.stem
        entry = {
            "run_id": run_id,
            "scene": match.group("scene_name"),
            "family": match.group("family"),
            "arm": match.group("arm"),
            "seed": int(match.group("seed")),
            "source": str(json_path),
        }
        for field in FIELDS:
            if field in record:
                entry[field] = record[field]
        curve = record.get("eval_curve")
        if isinstance(curve, list):
            entry["eval_curve"] = [
                {k: pt.get(k) for k in ("step", "wall_s", "psnr", "ssim", "lpips", "n_gaussians")}
                for pt in curve
            ]
            entry["eval_points"] = len(curve)
        runs[run_id] = entry

    # Pre-registered time-to-target-quality (pd vs paired ctrl final PSNR).
    time_to_target: dict[str, dict] = {}
    ctrl_final = {
        (r["scene"], r["seed"]): r["psnr"]
        for r in runs.values() if r.get("arm") == "ctrl" and "psnr" in r
    }
    for run_id, run in runs.items():
        if run.get("arm") != "pd":
            continue
        target = ctrl_final.get((run["scene"], run["seed"]))
        reached = None
        if target is not None:
            for pt in run.get("eval_curve", []):
                if pt.get("psnr") is not None and pt["psnr"] >= target:
                    reached = {"step": pt["step"], "wall_s": pt.get("wall_s")}
                    break
        time_to_target[run_id] = {
            "scene": run["scene"], "seed": run["seed"],
            "target_psnr": target, "reached": reached,
        }

    summary = {
        "tool": "collect_confirmatory_results",
        "n_runs": len(runs),
        "n_ctrl": sum(1 for r in runs.values() if r.get("arm") == "ctrl"),
        "n_pd": sum(1 for r in runs.values() if r.get("arm") == "pd"),
        "runs": runs,
        "time_to_target": time_to_target,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = collect(args.in_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
