#!/usr/bin/env python
"""Generate temporal-fidelity rows from a completed renderer session."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scripts.analyze_temporal_sequence import analyze  # noqa: E402
from schema_validation import validate_schema  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(source_root: Path, session: dict) -> list[dict]:
    suite = _load(source_root / "benchmark" / "suite.json")
    cases = {row["case_id"]: row for row in suite["cases"]}
    return [
        {
            "case_id": item["case_id"],
            "renderer": item["renderer"],
            "metrics": str(source_root / item["metrics_path"]),
            "ground_truth": str(source_root / cases[item["case_id"]]["ground_truth_path"]),
        }
        for item in session["completed"]
    ]


def run(source_root: Path, session_path: Path, output_root: Path, report_output: Path) -> dict:
    session = _load(session_path)
    if session.get("status") != "complete":
        raise ValueError("renderer session must be complete before temporal analysis")
    plan = build_plan(source_root, session)
    schema = _load(ROOT / "benchmark" / "schemas" / "temporal-result.schema.json")
    rows = []
    for index, item in enumerate(plan, start=1):
        print(f"[{index:02d}/{len(plan)}] {item['case_id']} :: {item['renderer']}", flush=True)
        result = analyze(Path(item["metrics"]), Path(item["ground_truth"]))
        validate_schema(result, schema)
        result_id = result["source_result"]["result_id"]
        destination = output_root / item["renderer"] / item["case_id"] / f"{result_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(result)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_result"]["config_id"], []).append(row)
    aggregates = []
    for config_id, config_rows in sorted(grouped.items()):
        aggregates.append({
            "config_id": config_id,
            "case_count": len(config_rows),
            "mean_temporal_residual": statistics.mean(
                row["metrics"]["mean_temporal_residual"] for row in config_rows
            ),
            "mean_luma_temporal_residual": statistics.mean(
                row["metrics"]["mean_luma_temporal_residual"] for row in config_rows
            ),
            "mean_temporal_delta_psnr_db": statistics.mean(
                row["metrics"]["temporal_delta_psnr_db"] for row in config_rows
            ),
        })
    report = {
        "schema_version": "1.0", "status": "complete",
        "source_session": str(session_path), "row_count": len(rows),
        "aggregates": aggregates,
    }
    report_output.mkdir(parents=True, exist_ok=True)
    (report_output / "temporal-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# EPIC-05 ordered-camera temporal fidelity", "",
        "This CPU analysis uses retained Tier A PNG evidence. It measures adjacent-frame",
        "RGB-delta residual against GT and does not claim motion-compensated video quality.", "",
        "| Config | Cases | Mean residual | Mean luma residual | Temporal delta PSNR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['config_id']} | {row['case_count']} | "
            f"{row['mean_temporal_residual']:.6f} | "
            f"{row['mean_luma_temporal_residual']:.6f} | "
            f"{row['mean_temporal_delta_psnr_db']:.3f} dB |"
        )
    (report_output / "temporal-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "measured-temporal")
    parser.add_argument("--report-output", type=Path, default=ROOT / "reports" / "generated" / "temporal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    session = _load(args.session.resolve())
    if args.dry_run:
        print(json.dumps(build_plan(args.source_root.resolve(), session), indent=2))
        return 0
    report = run(
        args.source_root.resolve(), args.session.resolve(),
        args.output_root.resolve(), args.report_output.resolve(),
    )
    print(json.dumps({"status": report["status"], "row_count": report["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
