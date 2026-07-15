"""Generate reproducible leaderboard artifacts from benchmark JSON files."""
from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping

from analysis.pareto import pareto_analysis
from analysis.efficiency import calculate_efficiency_score
from analysis.visualization import export_pareto_html
from benchmark_suite import BENCHMARK_SUITE_VERSION, official_metadata_matches_suite


PSNR_THRESHOLDS = (30, 31, 32)
SUITE_IDENTITY_KEYS = (
    "suite_id", "suite_case_id", "dataset_sha256", "scene_sha256",
    "camera_sha256", "resolution_profile", "resolution", "hardware",
)


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _quality_status(source):
    quality = source.get("quality", {}) if isinstance(source.get("quality"), dict) else {}
    status = source.get("quality_status", quality.get("status"))
    if status is None and "passed" in quality:
        status = "passed" if quality["passed"] else "failed"
    return status


def _common_metadata(document: Mapping, source_path: str) -> dict:
    environment = document.get("environment", {}) if isinstance(document, dict) else {}
    protocol = document.get("protocol", {}) if isinstance(document, dict) else {}
    suite = document.get("benchmark_suite", {})
    if not isinstance(suite, dict):
        suite = {}
    return {
        "source_file": source_path,
        "hardware": {
            "gpu": environment.get("gpu"),
            "driver": environment.get("driver"),
            "driver_cuda": environment.get("driver_cuda"),
            "cuda_toolkit": environment.get("cuda_toolkit"),
            "pytorch": environment.get("pytorch"),
        },
        "protocol": protocol,
        "suite": suite,
    }


def _record_from_speed(source: Mapping, metadata: Mapping, benchmark_type: str) -> dict:
    quality = source.get("quality", {}) if isinstance(source.get("quality"), dict) else {}
    suite = metadata.get("suite", {})
    resolution = suite.get("resolution", metadata.get("protocol", {}).get("resolution"))
    official_eligible = official_metadata_matches_suite(suite)
    return {
        "renderer": _first(source.get("renderer"), source.get("renderer_name")),
        "benchmark_type": source.get("benchmark_type", benchmark_type),
        "cohort": source.get("cohort"),
        "gaussians": source.get("gaussians", source.get("num_gaussians")),
        "fps": _first(source.get("fps"), source.get("mean_fps"), source.get("fps_from_mean")),
        "latency_ms": _first(
            source.get("latency_ms"),
            source.get("mean_latency_ms"),
            source.get("mean_ms"),
            source.get("mean_gpu_ms"),
        ),
        "p99_latency_ms": _first(
            source.get("p99_latency_ms"), source.get("p99_ms"), source.get("p99_gpu_ms")
        ),
        "peak_vram_mb": source.get("peak_vram_mb"),
        "psnr": _first(source.get("psnr"), source.get("psnr_vs_gt"), quality.get("mean_psnr_db")),
        "ssim": _first(source.get("ssim"), source.get("ssim_vs_gt"), quality.get("mean_ssim")),
        "lpips": _first(source.get("lpips"), source.get("lpips_vs_gt"), quality.get("mean_lpips")),
        "quality_status": _quality_status(source),
        "effective_fps": source.get("effective_fps"),
        "quality_factor": source.get("quality_factor"),
        "benchmark_suite_version": source.get("benchmark_suite_version", BENCHMARK_SUITE_VERSION),
        "official_eligible": official_eligible,
        "official": suite.get("official"),
        "validated": suite.get("validated"),
        "suite_id": suite.get("suite_id"),
        "suite_version": suite.get("suite_version"),
        "suite_case_id": suite.get("suite_case_id"),
        "dataset_sha256": suite.get("dataset_sha256"),
        "scene_sha256": suite.get("scene_sha256"),
        "camera_sha256": suite.get("camera_sha256"),
        "resolution_profile": suite.get("resolution_profile"),
        "resolution": resolution,
        "hardware": metadata.get("hardware", {}).get("gpu"),
        "metadata": dict(metadata),
    }


def normalize_benchmark_document(document: Mapping, source_path: str) -> list[dict]:
    """Normalize supported historical and current JSON formats to flat rows."""
    metadata = _common_metadata(document, source_path)
    status = str(document.get("status", ""))
    default_type = document.get("benchmark_type")
    if default_type is None and "synthetic" in status:
        default_type = "synthetic_stress"

    records: list[dict] = []
    if isinstance(document.get("records"), list):
        records.extend(_record_from_speed(row, metadata, default_type or "real_scene_speed") for row in document["records"])
    elif isinstance(document.get("results"), list):
        records.extend(_record_from_speed(row, metadata, default_type or "real_scene_speed") for row in document["results"])
    elif isinstance(document.get("results"), dict):
        records.extend(_record_from_speed(row, metadata, row.get("benchmark_type", default_type or "real_scene_speed")) for row in document["results"].values())
    elif isinstance(document, dict) and all(isinstance(value, dict) for value in document.values()):
        for name, row in document.items():
            row = {**row, "renderer": row.get("renderer", name)}
            records.append(_record_from_speed(row, metadata, row.get("benchmark_type", default_type or "real_scene_speed")))

    if isinstance(document.get("quality"), list):
        for row in document["quality"]:
            records.append({
                "renderer": row.get("renderer"),
                "benchmark_type": "real_scene_quality",
                "cohort": document.get("reference", {}).get("scene"),
                "gaussians": None,
                "fps": None,
                "latency_ms": None,
                "p99_latency_ms": None,
                "peak_vram_mb": None,
                "psnr": row.get("mean_psnr_db"),
                "ssim": row.get("mean_ssim"),
                "lpips": row.get("mean_lpips"),
                "quality_status": "measured",
                "effective_fps": None,
                "quality_factor": None,
                "benchmark_suite_version": document.get("benchmark_suite_version", BENCHMARK_SUITE_VERSION),
                "metadata": dict(metadata),
            })

    if isinstance(document.get("speed_smoke"), dict):
        smoke = document["speed_smoke"]
        for row in smoke.get("results", []):
            records.append(_record_from_speed(
                row,
                {**metadata, "protocol": {**metadata.get("protocol", {}), "speed_smoke": {
                    "camera_resolution": smoke.get("camera_resolution"),
                    "frames": smoke.get("frames"),
                    "warmup_frames": smoke.get("warmup_frames"),
                    "repeats": smoke.get("repeats"),
                }}},
                "real_scene_speed",
            ))

    return [record for record in records if record.get("renderer")]


def load_records(paths: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        records.extend(normalize_benchmark_document(document, path))
    return records


def _sort_speed(records):
    return sorted(
        [record for record in records if record.get("fps") is not None],
        key=lambda record: (-float(record["fps"]), record["renderer"]),
    )


def _sort_quality(records):
    eligible = [
        record for record in records
        if record.get("benchmark_type") != "synthetic_stress"
        and record.get("psnr") is not None
        and record.get("ssim") is not None
        and record.get("lpips") is not None
    ]
    return sorted(
        eligible,
        key=lambda record: (-float(record["psnr"]), -float(record["ssim"]), float(record["lpips"]), record["renderer"]),
    )


def _sort_memory(records):
    return sorted(
        [record for record in records if record.get("peak_vram_mb") is not None],
        key=lambda record: (float(record["peak_vram_mb"]), record["renderer"]),
    )


def _compact(record: Mapping) -> dict:
    keys = (
        "renderer", "benchmark_type", "cohort", "gaussians", "fps", "latency_ms",
        "p99_latency_ms", "peak_vram_mb", "psnr", "ssim", "lpips",
        "quality_status", "effective_fps", "quality_factor",
        "efficiency_score", "benchmark_suite_version", "suite_id",
        "suite_case_id", "resolution_profile", "resolution", "hardware",
        "cohort_count",
    )
    return {key: record.get(key) for key in keys if record.get(key) is not None}


def _cohort_key(record: Mapping) -> tuple:
    values = []
    for key in SUITE_IDENTITY_KEYS:
        value = record.get(key)
        values.append(tuple(value) if isinstance(value, list) else value)
    return tuple(values)


def _merge_official_records(records: Iterable[Mapping]) -> list[dict]:
    """Join speed and quality artifacts only when every suite identity matches."""
    merged = {}
    metric_keys = (
        "fps", "latency_ms", "p99_latency_ms", "peak_vram_mb",
        "psnr", "ssim", "lpips", "quality_status",
    )
    for source in records:
        if not source.get("official_eligible") or not official_metadata_matches_suite(source):
            continue
        key = (source["renderer"], _cohort_key(source))
        target = merged.setdefault(key, {
            **{name: source.get(name) for name in SUITE_IDENTITY_KEYS},
            "renderer": source["renderer"],
            "benchmark_type": "official_efficiency",
            "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
            "official_eligible": True,
            "official": True,
            "validated": True,
            "suite_version": source.get("suite_version"),
        })
        for name in metric_keys:
            value = source.get(name)
            if value is None:
                continue
            existing = target.get(name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"Conflicting official {name} values for {source['renderer']} "
                    f"in {source.get('suite_case_id')}"
                )
            target[name] = value
    for record in merged.values():
        if record.get("fps") is not None and record.get("psnr") is not None:
            record["efficiency_score"] = calculate_efficiency_score(
                record["fps"], record["psnr"]
            )
    return list(merged.values())


def _geometric_mean(values) -> float:
    values = [float(value) for value in values]
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _aggregate_suite(records: list[dict]) -> list[dict]:
    """Aggregate only renderers covering every submitted official cohort."""
    cohorts = {_cohort_key(record) for record in records}
    by_renderer = {}
    for record in records:
        by_renderer.setdefault(record["renderer"], []).append(record)
    aggregated = []
    for renderer, rows in by_renderer.items():
        if {_cohort_key(row) for row in rows} != cohorts:
            continue
        if any(row.get("fps") is None or row.get("psnr") is None for row in rows):
            continue
        aggregated.append({
            "renderer": renderer,
            "benchmark_type": "official_suite_aggregate",
            "fps": _geometric_mean(row["fps"] for row in rows),
            "latency_ms": _geometric_mean(row["latency_ms"] for row in rows)
            if all(row.get("latency_ms") for row in rows) else None,
            "p99_latency_ms": max(
                (row["p99_latency_ms"] for row in rows if row.get("p99_latency_ms") is not None),
                default=None,
            ),
            "peak_vram_mb": max(
                (row["peak_vram_mb"] for row in rows if row.get("peak_vram_mb") is not None),
                default=None,
            ),
            "psnr": min(row["psnr"] for row in rows),
            "ssim": min((row["ssim"] for row in rows if row.get("ssim") is not None), default=None),
            "lpips": max((row["lpips"] for row in rows if row.get("lpips") is not None), default=None),
            "efficiency_score": _geometric_mean(row["efficiency_score"] for row in rows),
            "cohort_count": len(cohorts),
            "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        })
    return aggregated


def _quality_constrained(records: list[dict]) -> dict:
    return {
        str(threshold): [
            _compact(record) for record in sorted(
                [
                    record for record in records
                    if record.get("fps") is not None
                    and record.get("psnr") is not None
                    and float(record["psnr"]) >= threshold
                ],
                key=lambda record: (-float(record["fps"]), record["renderer"]),
            )
        ]
        for threshold in PSNR_THRESHOLDS
    }


def _boards(records: list[dict]) -> tuple[dict, dict]:
    pareto = pareto_analysis(records)
    by_name = {record["renderer"]: record for record in records}
    boards = {
        "speed": [_compact(record) for record in _sort_speed(records)],
        "quality": [_compact(record) for record in _sort_quality(records)],
        "memory": [_compact(record) for record in _sort_memory(records)],
        "quality_constrained": _quality_constrained(records),
        "efficiency": [
            _compact(record) for record in sorted(
                [
                    record for record in records
                    if record.get("efficiency_score") is not None
                    and float(record["psnr"]) >= min(PSNR_THRESHOLDS)
                ],
                key=lambda record: (-float(record["efficiency_score"]), record["renderer"]),
            )
        ],
        "pareto": [_compact(by_name[name]) for name in pareto["frontier"] if name in by_name],
    }
    return boards, pareto


def generate_leaderboard(records: Iterable[Mapping]) -> dict:
    records = [dict(record) for record in records]
    official_records = _merge_official_records(records)
    aggregate_records = _aggregate_suite(official_records)
    boards, pareto = _boards(aggregate_records)
    cohorts = []
    for cohort_key in sorted({_cohort_key(record) for record in official_records}, key=str):
        cohort_records = [record for record in official_records if _cohort_key(record) == cohort_key]
        cohort_boards, cohort_pareto = _boards(cohort_records)
        identity = {name: cohort_records[0].get(name) for name in SUITE_IDENTITY_KEYS}
        cohorts.append({
            "identity": identity,
            "leaderboards": cohort_boards,
            "pareto_analysis": cohort_pareto,
        })
    return {
        "schema_version": 1,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "generated_by": "src/leaderboard/generator.py",
        "official_only": True,
        "quality_constraints": {
            "metric": "psnr",
            "thresholds_db": list(PSNR_THRESHOLDS),
            "rule": "renderer must meet the threshold in every aggregated cohort",
        },
        "efficiency_metric": {
            "name": "quality_adjusted_fps",
            "formula": "fps * 10 ** ((psnr - 32) / 10)",
            "minimum_psnr_db": min(PSNR_THRESHOLDS),
            "aggregation": "geometric mean across complete official cohorts",
        },
        "leaderboards": boards,
        "cohorts": cohorts,
        "pareto_analysis": pareto,
        "source_record_count": len(records),
        "official_record_count": len(official_records),
        "excluded_unofficial_count": len(records) - sum(
            1 for record in records
            if record.get("official_eligible") and official_metadata_matches_suite(record)
        ),
    }


def _markdown_table(title: str, rows: list[Mapping], columns: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["No eligible rows.", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "N/A")) for column in columns) + " |")
    lines.append("")
    return lines


def render_markdown(leaderboard: Mapping) -> str:
    boards = leaderboard["leaderboards"]
    lines = [
        "# 3DGS Renderer Leaderboard",
        "",
        "Official rankings include only hash-validated benchmark-suite records. Overall rows require complete coverage of every submitted cohort.",
        "",
    ]
    for threshold in PSNR_THRESHOLDS:
        lines += _markdown_table(
            f"Fastest @ PSNR >= {threshold}",
            boards["quality_constrained"][str(threshold)],
            ["renderer", "fps", "psnr", "efficiency_score", "cohort_count"],
        )
    lines += _markdown_table("Efficiency Score", boards["efficiency"], ["renderer", "efficiency_score", "fps", "psnr", "cohort_count"])
    lines += _markdown_table("Memory Leaderboard", boards["memory"], ["renderer", "peak_vram_mb", "cohort_count"])
    lines += _markdown_table("Pareto Leaderboard", boards["pareto"], ["renderer", "fps", "psnr", "ssim", "lpips"])
    return "\n".join(lines).strip() + "\n"


def render_html(leaderboard: Mapping) -> str:
    markdown = html.escape(render_markdown(leaderboard))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3DGS Renderer Leaderboard</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:auto;padding:24px;line-height:1.5}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:16px;border-radius:8px}}</style>
</head><body><pre>{markdown}</pre></body></html>
"""


def write_leaderboard(leaderboard: Mapping, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "leaderboard.json"), "w", encoding="utf-8") as handle:
        json.dump(leaderboard, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with open(os.path.join(output_dir, "leaderboard.md"), "w", encoding="utf-8") as handle:
        handle.write(render_markdown(leaderboard))
    with open(os.path.join(output_dir, "leaderboard.html"), "w", encoding="utf-8") as handle:
        handle.write(render_html(leaderboard))
    visualization_records = leaderboard["leaderboards"]["speed"]
    export_pareto_html(
        visualization_records,
        leaderboard["pareto_analysis"],
        os.path.join(output_dir, "quality_speed.html"),
        title="3DGS Quality-Speed Efficiency",
    )

