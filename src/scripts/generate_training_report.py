#!/usr/bin/env python
"""Aggregate the complete native-training matrix into compact JSON and Markdown."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from schema_validation import validate_schema  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(root: Path, input_root: Path) -> dict:
    config = _load(root / "benchmark" / "training.json")
    schema = _load(root / "benchmark" / "schemas" / "training-result.schema.json")
    rows = []
    for path in sorted(input_root.glob("*/*/metrics.json")):
        row = _load(path)
        validate_schema(row, schema)
        if row["status"] == "complete":
            rows.append(row)
    expected = {
        (backend["id"], case["case_id"])
        for backend in config["backends"] for case in config["cases"]
    }
    actual = {(row["backend"]["id"], row["case"]["case_id"]) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        print("WARNING: training matrix incomplete:", missing, file=__import__("sys").stderr)
    summaries = []
    for backend in config["backends"]:
        group = [row for row in rows if row["backend"]["id"] == backend["id"]]
        summaries.append({
            "backend": backend["id"], "case_count": len(group),
            "mean_wall_time_seconds": statistics.mean(
                row["performance"]["wall_time_seconds"] for row in group),
            "mean_iterations_per_second": statistics.mean(
                row["performance"]["iterations_per_second"] for row in group),
            "max_peak_process_gpu_memory_mib": max(
                row["performance"]["peak_process_gpu_memory_mib"] for row in group),
            "mean_psnr_db": statistics.mean(row["quality"]["mean_psnr_db"] for row in group if row.get("quality") and row["quality"].get("mean_psnr_db") is not None) if any(row.get("quality") for row in group) else None,
            "mean_ssim": statistics.mean(row["quality"]["mean_ssim"] for row in group if row.get("quality") and row["quality"].get("mean_ssim") is not None) if any(row.get("quality") for row in group) else None,
            "mean_lpips": statistics.mean(row["quality"]["mean_lpips"] for row in group if row.get("quality") and row["quality"].get("mean_lpips") is not None) if any(row.get("quality") for row in group) else None,
        })
    return {"schema_version": "1.0", "status": "complete", "track": "native_training",
            "row_count": len(rows), "summaries": summaries, "results": rows}


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "training-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EPIC-05 native training matrix", "",
        "Fixed 30,000-iteration native runs; this cohort is not mixed with the common-checkpoint renderer ranking.", "",
        "Wall-time rows produced by the shard launcher use eight identical A100 GPUs concurrently and are labeled `concurrent_8_gpu_sharded` in provenance.", "",
        "| Backend | Cases | Mean wall time | Iterations/s | Peak VRAM | PSNR | SSIM | LPIPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summaries"]:
        lines.append(
            f"| {row['backend']} | {row['case_count']} | {row['mean_wall_time_seconds']:.1f} s | "
            f"{row['mean_iterations_per_second']:.2f} | {row['max_peak_process_gpu_memory_mib']:.0f} MiB | "
            f"{row['mean_psnr_db']:.3f} | {row['mean_ssim']:.5f} | {row['mean_lpips']:.5f} |"
        )
    (output / "training-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input-root", type=Path, default=ROOT / "artifacts" / "training")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "generated" / "training")
    args = parser.parse_args(argv)
    report = aggregate(args.root.resolve(), args.input_root.resolve())
    write_report(report, args.output.resolve())
    print(json.dumps({"status": report["status"], "row_count": report["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
