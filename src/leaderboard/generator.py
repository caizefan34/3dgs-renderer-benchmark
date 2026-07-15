"""Generate reproducible leaderboard artifacts from benchmark JSON files."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from analysis.pareto import pareto_analysis
from benchmark_suite import BENCHMARK_SUITE_VERSION


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
    }


def _record_from_speed(source: Mapping, metadata: Mapping, benchmark_type: str) -> dict:
    quality = source.get("quality", {}) if isinstance(source.get("quality"), dict) else {}
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
        "benchmark_suite_version",
    )
    return {key: record.get(key) for key in keys if record.get(key) is not None}


def generate_leaderboard(records: Iterable[Mapping]) -> dict:
    records = [dict(record) for record in records]
    pareto = pareto_analysis(records)
    by_name = {record["renderer"]: record for record in records}
    return {
        "schema_version": 1,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "generated_by": "src/leaderboard/generator.py",
        "leaderboards": {
            "speed": [_compact(record) for record in _sort_speed(records)],
            "quality": [_compact(record) for record in _sort_quality(records)],
            "memory": [_compact(record) for record in _sort_memory(records)],
            "pareto": [_compact(by_name[name]) for name in pareto["frontier"] if name in by_name],
        },
        "pareto_analysis": pareto,
        "source_record_count": len(records),
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
        "Generated from committed benchmark JSON artifacts. Synthetic stress speed rows remain separate from GT-quality rankings.",
        "",
    ]
    lines += _markdown_table("Speed Leaderboard", boards["speed"], ["renderer", "benchmark_type", "gaussians", "fps", "p99_latency_ms"])
    lines += _markdown_table("Quality Leaderboard", boards["quality"], ["renderer", "benchmark_type", "psnr", "ssim", "lpips"])
    lines += _markdown_table("Memory Leaderboard", boards["memory"], ["renderer", "benchmark_type", "gaussians", "peak_vram_mb"])
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

