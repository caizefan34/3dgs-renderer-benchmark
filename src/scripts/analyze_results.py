#!/usr/bin/env python
"""Generate quality-adjusted, Pareto, recommendation, and visual artifacts."""
import argparse
from dataclasses import asdict
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.efficiency import (  # noqa: E402
    QualityAdjustmentConfig,
    calculate_quality_adjusted_efficiency,
)
from analysis.pareto import pareto_analysis  # noqa: E402
from analysis.recommendations import build_recommendations  # noqa: E402
from analysis.visualization import export_pareto_html  # noqa: E402


def normalize_records(data):
    """Normalize the standard renderer-keyed output or an explicit row list."""
    default_type = data.get("benchmark_type") if isinstance(data, dict) else None
    if isinstance(data, list):
        items = [(None, row) for row in data]
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        items = [(None, row) for row in data["records"]]
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        items = [(None, row) for row in data["results"]]
    elif isinstance(data, dict) and isinstance(data.get("results"), dict):
        items = list(data["results"].items())
    elif isinstance(data, dict):
        items = list(data.items())
    else:
        raise ValueError("Input must be a renderer-keyed object or a records list")

    records = []
    for fallback_name, source in items:
        if not isinstance(source, dict):
            continue
        quality = source.get("quality", {})
        renderer = source.get("renderer") or source.get("renderer_name") or fallback_name
        if not renderer:
            raise ValueError("Every record needs a renderer identifier")
        record = {
            "renderer": renderer,
            "fps": source.get("fps", source.get("mean_fps", source.get("fps_from_mean"))),
            "latency_ms": source.get("latency_ms", source.get("mean_latency_ms", source.get("mean_ms"))),
            "p99_latency_ms": source.get("p99_latency_ms", source.get("p99_ms")),
            "psnr": source.get("psnr", source.get("psnr_vs_gt", quality.get("mean_psnr_db"))),
            "ssim": source.get("ssim", source.get("ssim_vs_gt", quality.get("mean_ssim"))),
            "lpips": source.get("lpips", source.get("lpips_vs_gt", quality.get("mean_lpips"))),
            "peak_vram_mb": source.get("peak_vram_mb"),
            "stability_score": source.get("stability_score"),
            "quality_factor": source.get("quality_factor"),
            "effective_fps": source.get("effective_fps"),
            "benchmark_type": source.get("benchmark_type", default_type),
            "quality_status": source.get("quality_status", quality.get("status")),
        }
        if record["quality_status"] is None and "passed" in quality:
            record["quality_status"] = "passed" if quality["passed"] else "failed"
        if record["stability_score"] is None:
            median = source.get("median_latency_ms", source.get("median_ms"))
            p99 = record["p99_latency_ms"]
            if median is not None and p99:
                record["stability_score"] = min(float(median) / float(p99), 1.0)
        if record["benchmark_type"] == "synthetic_stress":
            record.update(psnr=None, ssim=None, lpips=None)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-renderer")
    parser.add_argument("--psnr-weight", type=float, default=0.25)
    parser.add_argument("--ssim-weight", type=float, default=25.0)
    parser.add_argument("--lpips-weight", type=float, default=10.0)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        records = normalize_records(json.load(handle))
    types = {record["benchmark_type"] for record in records if record["benchmark_type"]}
    if len(types) > 1:
        raise ValueError("Analyze one benchmark taxonomy/cohort at a time")

    if args.reference_renderer:
        by_name = {record["renderer"]: record for record in records}
        if args.reference_renderer not in by_name:
            raise ValueError("Reference renderer is not present in the input")
        reference = by_name[args.reference_renderer]
        config = QualityAdjustmentConfig(
            psnr_drop_weight=args.psnr_weight,
            ssim_drop_weight=args.ssim_weight,
            lpips_increase_weight=args.lpips_weight,
        )
        quality_adjustment = {"formula_id": config.formula_id, "coefficients": asdict(config)}
        for record in records:
            adjusted = calculate_quality_adjusted_efficiency(
                record["fps"] or 0.0, record, reference, config
            )
            record["quality_factor"] = adjusted.quality_factor
            record["effective_fps"] = adjusted.effective_fps
    else:
        quality_adjustment = None

    pareto = pareto_analysis(records)
    recommendations = build_recommendations(records, pareto["frontier"])
    os.makedirs(args.output_dir, exist_ok=True)
    artifacts = {
        "pareto_frontier.json": pareto,
        "recommendations.json": recommendations,
        "evaluation_records.json": {
            "schema_version": 1,
            "experimental_metrics": True,
            "quality_adjustment": quality_adjustment,
            "records": records,
        },
    }
    for filename, content in artifacts.items():
        with open(os.path.join(args.output_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False, allow_nan=False)
    export_pareto_html(
        records, pareto, os.path.join(args.output_dir, "pareto_frontier.html")
    )


if __name__ == "__main__":
    main()
