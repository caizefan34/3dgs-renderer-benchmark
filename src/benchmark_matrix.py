"""Tier-safe aggregation and Pareto analysis for Benchmark Matrix v2."""
from __future__ import annotations

import json
import math
import statistics
import csv
import html
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


TIERS = ("measured", "reproduced", "paper_reported")
REQUIRED_PERFORMANCE = (
    "fps",
    "fps_ci95_low",
    "fps_ci95_high",
    "frame_time_ms",
    "p95_frame_time_ms",
    "p99_frame_time_ms",
    "peak_vram_mb",
    "startup_time_ms",
    "renderer_init_time_ms",
    "scene_load_time_ms",
    "renderer_prepare_time_ms",
    "time_to_first_frame_ms",
)
REQUIRED_QUALITY = ("psnr_db", "ssim", "lpips")


class MatrixValidationError(ValueError):
    """Raised when a v2 result is not scientifically rankable."""


def _require(mapping: Mapping, keys: Iterable[str], path: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise MatrixValidationError(f"{path}: missing {', '.join(missing)}")


def _require_sha256(value, path: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise MatrixValidationError(f"{path}: expected lowercase SHA-256")


def validate_result(document: Mapping) -> None:
    """Validate semantic requirements that JSON Schema alone cannot express."""
    _require(
        document,
        ("schema_version", "result_id", "evidence_tier", "status", "renderer",
         "benchmark", "environment", "metrics", "provenance"),
        "$",
    )
    if document["schema_version"] != "2.0":
        raise MatrixValidationError("$.schema_version: expected '2.0'")
    tier = document["evidence_tier"]
    if tier not in TIERS:
        raise MatrixValidationError(f"$.evidence_tier: unsupported tier {tier!r}")

    renderer = document["renderer"]
    _require(
        renderer,
        ("id", "config_id", "name", "version", "source_uri", "source_commit",
         "build_command", "runtime_command", "api", "backend", "platforms", "features"),
        "$.renderer",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", str(renderer["source_commit"])):
        raise MatrixValidationError("$.renderer.source_commit: expected pinned 40-character git commit")

    benchmark = document["benchmark"]
    _require(
        benchmark,
        ("suite_id", "suite_version", "track_id", "case_id", "dataset_id",
         "dataset_sha256", "scene_id", "scene_tier", "checkpoint_sha256",
         "gaussian_count", "sh_degree", "camera_trajectory_id",
         "camera_trajectory_sha256", "quality_reference_sha256", "resolution",
         "color_space", "background", "protocol_id", "protocol_sha256"),
        "$.benchmark",
    )
    _require(benchmark["resolution"], ("width", "height"), "$.benchmark.resolution")
    for key in ("dataset_sha256", "checkpoint_sha256", "camera_trajectory_sha256", "quality_reference_sha256", "protocol_sha256"):
        _require_sha256(benchmark[key], f"$.benchmark.{key}")

    environment = document["environment"]
    _require(
        environment,
        ("hardware_profile_id", "gpu", "gpu_uuid", "gpu_vram_mb", "cpu", "ram_mb", "os",
         "driver", "cuda", "python", "pytorch", "benchmark_commit", "clock_policy", "power_limit_w"),
        "$.environment",
    )
    for key in ("gpu_vram_mb", "ram_mb"):
        if not math.isfinite(float(environment[key])) or float(environment[key]) <= 0:
            raise MatrixValidationError(f"$.environment.{key}: must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", str(environment["benchmark_commit"])):
        raise MatrixValidationError("$.environment.benchmark_commit: expected 40-character git commit")

    provenance = document["provenance"]
    _require(provenance, ("source_type", "source_uri", "measured_at", "raw_samples_uri", "raw_samples_sha256"), "$.provenance")
    expected_source = {
        "measured": "repository_run",
        "reproduced": "official_implementation_run",
        "paper_reported": "paper",
    }[tier]
    if provenance["source_type"] != expected_source:
        raise MatrixValidationError(
            f"$.provenance.source_type: tier {tier!r} requires {expected_source!r}"
        )
    if tier == "paper_reported":
        _require(provenance, ("citation", "paper_table_or_figure"), "$.provenance")
    elif not provenance["raw_samples_uri"] or not provenance["raw_samples_sha256"]:
        raise MatrixValidationError("measured and reproduced results require hashed raw samples")
    if provenance["raw_samples_sha256"] is not None:
        _require_sha256(provenance["raw_samples_sha256"], "$.provenance.raw_samples_sha256")
    if document["status"] != "complete":
        if document["metrics"] is not None and not isinstance(document["metrics"], Mapping):
            raise MatrixValidationError("$.metrics: expected object or null for unsuccessful attempt")
        return

    metrics = document["metrics"]
    if not isinstance(metrics, Mapping):
        raise MatrixValidationError("$.metrics: complete result requires metrics")
    _require(metrics, ("performance", "quality", "raw_samples"), "$.metrics")
    _require(metrics["performance"], REQUIRED_PERFORMANCE, "$.metrics.performance")
    _require(metrics["quality"], REQUIRED_QUALITY, "$.metrics.quality")
    for key in REQUIRED_PERFORMANCE:
        if metrics["performance"][key] is None:
            raise MatrixValidationError(f"$.metrics.performance.{key}: null is not rankable")
        if not math.isfinite(float(metrics["performance"][key])) or float(metrics["performance"][key]) <= 0:
            raise MatrixValidationError(f"$.metrics.performance.{key}: must be positive")
    for key in REQUIRED_QUALITY:
        if metrics["quality"][key] is None:
            raise MatrixValidationError(f"$.metrics.quality.{key}: null is not rankable")
        if not math.isfinite(float(metrics["quality"][key])):
            raise MatrixValidationError(f"$.metrics.quality.{key}: must be finite")
    if not 0 <= float(metrics["quality"]["ssim"]) <= 1:
        raise MatrixValidationError("$.metrics.quality.ssim: expected [0,1]")
    if float(metrics["quality"]["lpips"]) < 0:
        raise MatrixValidationError("$.metrics.quality.lpips: must be non-negative")


def load_results(paths: Iterable[str | Path]) -> tuple[list[dict], list[dict]]:
    """Load result files, returning (valid, rejected) without hiding failures."""
    valid, rejected = [], []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            validate_result(document)
            if document["status"] != "complete":
                raise MatrixValidationError(f"status is {document['status']!r}, not 'complete'")
            valid.append(document)
        except (OSError, json.JSONDecodeError, MatrixValidationError) as exc:
            rejected.append({"source_file": str(path), "reason": str(exc)})
    return valid, rejected


def cohort_key(document: Mapping) -> tuple:
    """Identity fields that must match before rows may share a table."""
    benchmark = document["benchmark"]
    environment = document["environment"]
    resolution = benchmark["resolution"]
    return (
        document["evidence_tier"],
        benchmark["suite_id"],
        benchmark["suite_version"],
        benchmark["track_id"],
        benchmark["case_id"],
        benchmark["dataset_id"],
        benchmark["scene_id"],
        benchmark["checkpoint_sha256"],
        benchmark["gaussian_count"],
        benchmark["sh_degree"],
        benchmark["camera_trajectory_sha256"],
        benchmark["quality_reference_sha256"],
        resolution["width"],
        resolution["height"],
        benchmark["protocol_sha256"],
        environment["hardware_profile_id"],
        environment["gpu"],
        environment["driver"],
        environment["cuda"],
    )


def suite_cohort_key(document: Mapping) -> tuple:
    """Identity shared by every case in one publishable overall matrix."""
    benchmark = document["benchmark"]
    environment = document["environment"]
    resolution = benchmark["resolution"]
    paper_source = document["provenance"].get("source_uri") if document["evidence_tier"] == "paper_reported" else None
    return (
        document["evidence_tier"], benchmark["suite_id"], benchmark["suite_version"],
        benchmark["track_id"], benchmark["protocol_sha256"],
        resolution["width"], resolution["height"], benchmark["color_space"], benchmark["background"],
        environment["hardware_profile_id"], environment["gpu"], environment["driver"],
        environment["gpu_uuid"], environment["gpu_vram_mb"], environment["cpu"],
        environment["ram_mb"], environment["os"], environment["cuda"],
        environment["python"], environment["pytorch"], environment["benchmark_commit"],
        environment["clock_policy"], environment["power_limit_w"], paper_source,
    )


def _geometric_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0 for value in values):
        raise MatrixValidationError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _quality_utility(psnr: float, ssim: float, lpips: float) -> float:
    """Bounded, published utility used only by the efficiency ranking."""
    q_psnr = min(1.0, max(0.0, (float(psnr) - 20.0) / 20.0))
    q_ssim = min(1.0, max(0.0, (float(ssim) - 0.8) / 0.2))
    q_lpips = min(1.0, max(0.0, (0.4 - float(lpips)) / 0.4))
    if 0.0 in (q_psnr, q_ssim, q_lpips):
        return 0.0
    return (q_psnr * q_ssim * q_lpips) ** (1.0 / 3.0)


def _row(document: Mapping) -> dict:
    performance = document["metrics"]["performance"]
    quality = document["metrics"]["quality"]
    renderer = document["renderer"]
    utility = _quality_utility(quality["psnr_db"], quality["ssim"], quality["lpips"])
    resource_cost = performance["frame_time_ms"] * (performance["peak_vram_mb"] / 1024.0)
    return {
        "competitor_id": renderer["config_id"],
        "renderer_id": renderer["id"],
        "renderer": renderer["name"],
        "renderer_version": renderer["version"],
        "renderer_config_id": renderer["config_id"],
        "renderer_source_commit": renderer["source_commit"],
        "api": renderer["api"],
        "backend": renderer["backend"],
        "platforms": renderer["platforms"],
        "features": renderer["features"],
        "case_id": document["benchmark"]["case_id"],
        "fps": performance["fps"],
        "fps_ci95_low": performance["fps_ci95_low"],
        "fps_ci95_high": performance["fps_ci95_high"],
        "frame_time_ms": performance["frame_time_ms"],
        "p95_frame_time_ms": performance["p95_frame_time_ms"],
        "p99_frame_time_ms": performance["p99_frame_time_ms"],
        "peak_vram_mb": performance["peak_vram_mb"],
        "startup_time_ms": performance["startup_time_ms"],
        "renderer_init_time_ms": performance["renderer_init_time_ms"],
        "scene_load_time_ms": performance["scene_load_time_ms"],
        "renderer_prepare_time_ms": performance["renderer_prepare_time_ms"],
        "time_to_first_frame_ms": performance["time_to_first_frame_ms"],
        "psnr_db": quality["psnr_db"],
        "ssim": quality["ssim"],
        "lpips": quality["lpips"],
        "quality_utility": utility,
        "efficiency_score": utility / resource_cost if resource_cost > 0 else None,
        "source_result_id": document["result_id"],
    }


def _dominates(candidate: Mapping, record: Mapping, objectives: Mapping[str, str]) -> bool:
    weak, strict = [], []
    for metric, direction in objectives.items():
        if direction == "max":
            weak.append(candidate[metric] >= record[metric])
            strict.append(candidate[metric] > record[metric])
        else:
            weak.append(candidate[metric] <= record[metric])
            strict.append(candidate[metric] < record[metric])
    return all(weak) and any(strict)


def pareto_frontier(rows: Iterable[Mapping], objectives: Mapping[str, str]) -> list[str]:
    rows = list(rows)
    return sorted(
        row["competitor_id"] for row in rows
        if not any(other is not row and _dominates(other, row, objectives) for other in rows)
    )


def _aggregate_renderer(rows: list[Mapping]) -> dict:
    first = rows[0]
    result = {
        "renderer_id": first["renderer_id"],
        "competitor_id": first["competitor_id"],
        "renderer": first["renderer"],
        "renderer_version": first["renderer_version"],
        "renderer_config_id": first["renderer_config_id"],
        "renderer_source_commit": first["renderer_source_commit"],
        "api": first["api"],
        "backend": first["backend"],
        "platforms": first["platforms"],
        "features": first["features"],
        "case_count": len(rows),
        "fps": _geometric_mean(row["fps"] for row in rows),
        "fps_ci95_low": _geometric_mean(row["fps_ci95_low"] for row in rows),
        "fps_ci95_high": _geometric_mean(row["fps_ci95_high"] for row in rows),
        "frame_time_ms": _geometric_mean(row["frame_time_ms"] for row in rows),
        "p95_frame_time_ms": max(row["p95_frame_time_ms"] for row in rows),
        "p99_frame_time_ms": max(row["p99_frame_time_ms"] for row in rows),
        "peak_vram_mb": max(row["peak_vram_mb"] for row in rows),
        "startup_time_ms": statistics.median(row["startup_time_ms"] for row in rows),
        "renderer_init_time_ms": statistics.median(row["renderer_init_time_ms"] for row in rows),
        "scene_load_time_ms": statistics.median(row["scene_load_time_ms"] for row in rows),
        "renderer_prepare_time_ms": statistics.median(row["renderer_prepare_time_ms"] for row in rows),
        "time_to_first_frame_ms": statistics.median(row["time_to_first_frame_ms"] for row in rows),
        "psnr_db": statistics.mean(row["psnr_db"] for row in rows),
        "ssim": statistics.mean(row["ssim"] for row in rows),
        "lpips": statistics.mean(row["lpips"] for row in rows),
    }
    result["quality_utility"] = _quality_utility(result["psnr_db"], result["ssim"], result["lpips"])
    resource_cost = result["frame_time_ms"] * (result["peak_vram_mb"] / 1024.0)
    result["efficiency_score"] = result["quality_utility"] / resource_cost if resource_cost > 0 else None
    return result


def _best_balance(rows: list[Mapping]) -> str | None:
    if not rows:
        return None
    axes = {"fps": "max", "psnr_db": "max", "ssim": "max", "lpips": "min"}
    normalized = []
    for row in rows:
        distance = 0.0
        for metric, direction in axes.items():
            values = [float(item[metric]) for item in rows]
            low, high = min(values), max(values)
            score = 1.0 if high == low else (float(row[metric]) - low) / (high - low)
            if direction == "min":
                score = 1.0 - score
            distance += (1.0 - score) ** 2
        normalized.append((math.sqrt(distance), row["competitor_id"]))
    return min(normalized)[1]


def _single_cohort_report(documents: list[Mapping], required_cases: set[str], reference_renderer_id: str) -> dict:
    cohort_rows = defaultdict(list)
    renderer_rows = defaultdict(list)
    for document in documents:
        row = _row(document)
        cohort_rows[cohort_key(document)].append(row)
        renderer_identity = (
            row["renderer_id"], row["renderer_config_id"],
            row["renderer_version"], row["renderer_source_commit"],
        )
        renderer_rows[renderer_identity].append(row)

    excluded = {}
    overall = []
    case_cohorts = defaultdict(set)
    for key in cohort_rows:
        case_cohorts[key[4]].add(key)
    incompatible_cases = sorted(case for case, keys in case_cohorts.items() if len(keys) != 1)
    for renderer_identity, rows in renderer_rows.items():
        competitor_id = renderer_identity[1]
        covered = {row["case_id"] for row in rows}
        missing = sorted(required_cases - covered)
        duplicates = len(rows) != len(covered)
        if missing or duplicates or incompatible_cases:
            excluded[competitor_id] = {
                "missing_cases": missing,
                "reason": (
                    f"multiple immutable cohorts for cases: {', '.join(incompatible_cases)}"
                    if incompatible_cases else "duplicate case rows" if duplicates else "incomplete suite coverage"
                ),
            }
            continue
        overall.append(_aggregate_renderer(rows))

    reference_candidates = [rows for identity, rows in renderer_rows.items() if identity[0] == reference_renderer_id]
    reference_by_case = {
        row["case_id"]: row for row in reference_candidates[0]
    } if len(reference_candidates) == 1 else {}
    if set(reference_by_case) == required_cases:
        for aggregate in overall:
            identity = next(
                key for key in renderer_rows
                if key[0] == aggregate["renderer_id"] and key[1] == aggregate["renderer_config_id"]
            )
            rows = renderer_rows[identity]
            aggregate["speed_index"] = _geometric_mean(
                row["fps"] / reference_by_case[row["case_id"]]["fps"] for row in rows
            )
            aggregate["delta_psnr_db"] = statistics.mean(
                row["psnr_db"] - reference_by_case[row["case_id"]]["psnr_db"] for row in rows
            )
            aggregate["delta_ssim"] = statistics.mean(
                row["ssim"] - reference_by_case[row["case_id"]]["ssim"] for row in rows
            )
            aggregate["delta_lpips"] = statistics.mean(
                row["lpips"] - reference_by_case[row["case_id"]]["lpips"] for row in rows
            )

    overall.sort(key=lambda row: row["competitor_id"])
    primary_objectives = {"fps": "max", "psnr_db": "max", "ssim": "max", "lpips": "min"}
    speed_psnr = {"fps": "max", "psnr_db": "max"}
    speed_lpips = {"fps": "max", "lpips": "min"}
    speed = sorted(overall, key=lambda row: (-row.get("speed_index", row["fps"]), row["competitor_id"]))
    quality = sorted(overall, key=lambda row: (-row["quality_utility"], row["competitor_id"]))
    efficiency = sorted(overall, key=lambda row: (-row["efficiency_score"], row["competitor_id"]))
    memory = sorted(overall, key=lambda row: (row["peak_vram_mb"], row["competitor_id"]))
    pareto_ids = pareto_frontier(overall, primary_objectives)
    by_id = {row["competitor_id"]: row for row in overall}

    cohorts = []
    for key, rows in sorted(cohort_rows.items(), key=lambda item: str(item[0])):
        cohorts.append({
            "cohort_key": list(key),
            "rows": sorted(rows, key=lambda row: row["competitor_id"]),
            "pareto": {
                "speed_psnr": pareto_frontier(rows, speed_psnr),
                "speed_lpips": pareto_frontier(rows, speed_lpips),
                "combined": pareto_frontier(rows, primary_objectives),
            },
        })

    return {
        "overall": overall,
        "real_time": speed,
        "quality": quality,
        "quality_rankings": {
            "psnr": sorted(overall, key=lambda row: (-row["psnr_db"], row["competitor_id"])),
            "ssim": sorted(overall, key=lambda row: (-row["ssim"], row["competitor_id"])),
            "lpips": sorted(overall, key=lambda row: (row["lpips"], row["competitor_id"])),
        },
        "efficiency": efficiency,
        "memory": memory,
        "pareto": [by_id[renderer_id] for renderer_id in pareto_ids],
        "pareto_frontiers": {
            "speed_psnr": pareto_frontier(overall, speed_psnr),
            "speed_lpips": pareto_frontier(overall, speed_lpips),
            "combined": pareto_ids,
        },
        "recommendations": {
            "highest_fps": speed[0]["competitor_id"] if speed else None,
            "highest_quality": quality[0]["competitor_id"] if quality else None,
            "lowest_vram": memory[0]["competitor_id"] if memory else None,
            "best_efficiency": efficiency[0]["competitor_id"] if efficiency else None,
            "best_balance": _best_balance([by_id[item] for item in pareto_ids]),
        },
        "excluded": excluded,
        "cohorts": cohorts,
    }


def _tier_report(documents: list[Mapping], required_cases: set[str], reference_renderer_id: str) -> dict:
    groups = defaultdict(list)
    for document in documents:
        groups[suite_cohort_key(document)].append(document)
    reports = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        reports.append({
            "suite_cohort_key": list(key),
            "report": _single_cohort_report(rows, required_cases, reference_renderer_id),
        })
    if len(reports) == 1:
        primary = dict(reports[0]["report"])
    else:
        primary = _single_cohort_report([], required_cases, reference_renderer_id)
        if reports:
            primary["excluded"]["_multiple_hardware_or_software_cohorts"] = {
                "reason": "select one suite cohort before publishing an overall ranking",
                "cohort_count": len(reports),
            }
    primary["suite_cohorts"] = reports
    return primary


def generate_matrix_report(documents: Iterable[Mapping], suite: Mapping) -> dict:
    """Generate rankings while keeping evidence tiers physically separate."""
    required_cases = {
        case["case_id"] for case in suite["cases"] if case.get("required", True)
    }
    by_tier = {tier: [] for tier in TIERS}
    rejected = []
    for document in documents:
        try:
            validate_result(document)
        except MatrixValidationError as exc:
            rejected.append({"result_id": document.get("result_id"), "reason": str(exc)})
            continue
        if document["status"] != "complete":
            rejected.append({"result_id": document["result_id"], "reason": "result is not complete"})
            continue
        benchmark = document["benchmark"]
        expected_resolution = suite["ranking"]["primary_resolution"]
        resolution = [benchmark["resolution"]["width"], benchmark["resolution"]["height"]]
        mismatch = None
        if benchmark["suite_id"] != suite["suite_id"] or benchmark["suite_version"] != suite["version"]:
            mismatch = "suite identity mismatch"
        elif benchmark["track_id"] != suite["primary_track"]:
            mismatch = "track mismatch"
        elif benchmark["case_id"] not in required_cases:
            mismatch = "case is not in the primary suite"
        elif benchmark["protocol_sha256"] != suite["protocol_sha256"]:
            mismatch = "protocol hash mismatch"
        elif resolution != expected_resolution:
            mismatch = "primary resolution mismatch"
        if mismatch:
            rejected.append({"result_id": document["result_id"], "reason": mismatch})
            continue
        by_tier[document["evidence_tier"]].append(document)
    tier_reports = {
        tier: _tier_report(
            by_tier[tier], required_cases, suite["ranking"]["reference_renderer_id"]
        )
        for tier in TIERS
    }
    preferred_tier = next((tier for tier in TIERS if tier_reports[tier]["overall"]), None)
    return {
        "schema_version": "2.0",
        "suite_id": suite["suite_id"],
        "suite_version": suite["version"],
        "tier_policy": "never_mix",
        "required_cases": sorted(required_cases),
        "evidence_precedence": list(TIERS),
        "preferred_tier": preferred_tier,
        "preferred_recommendations": tier_reports[preferred_tier]["recommendations"] if preferred_tier else {},
        "tiers": tier_reports,
        "rejected": rejected,
    }


def write_report(report: Mapping, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ranking.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Renderer Rankings",
        "",
        "Evidence tiers are intentionally separate. Empty tables mean that no complete, comparable matrix has been submitted.",
        "",
    ]
    for tier, title in (("measured", "Tier A: Measured"), ("reproduced", "Tier B: Reproduced"), ("paper_reported", "Tier C: Paper Reported")):
        lines.extend((f"## {title}", ""))
        tier_report = report["tiers"][tier]
        rows = tier_report["overall"]
        if not rows:
            cohort_reports = tier_report.get("suite_cohorts", [])
            if not cohort_reports:
                lines.extend(("No renderer has complete suite coverage.", ""))
                continue
            lines.extend(("Multiple hardware/software cohorts exist; they are shown separately.", ""))
            for index, cohort in enumerate(cohort_reports, 1):
                lines.extend((f"### Cohort {index}", ""))
                _append_markdown_rows(lines, cohort["report"]["overall"])
                _append_detailed_boards(lines, cohort["report"], "####")
            continue
        _append_markdown_rows(lines, rows)
        _append_detailed_boards(lines, tier_report, "###")
    (output / "ranking.md").write_text("\n".join(lines), encoding="utf-8")

    columns = [
        "tier", "suite_cohort", "ranking_type", "rank", "renderer_id", "renderer", "fps", "frame_time_ms", "psnr_db",
        "ssim", "lpips", "peak_vram_mb", "fps_ci95_low", "fps_ci95_high", "startup_time_ms",
        "renderer_init_time_ms", "scene_load_time_ms", "renderer_prepare_time_ms", "time_to_first_frame_ms",
        "p95_frame_time_ms", "p99_frame_time_ms", "efficiency_score", "case_count",
    ]
    with (output / "ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for tier in TIERS:
            cohort_reports = report["tiers"][tier].get("suite_cohorts", [])
            for index, cohort in enumerate(cohort_reports, 1):
                boards = _named_boards(cohort["report"])
                for ranking_type, rows in boards:
                    for rank, row in enumerate(rows, 1):
                        writer.writerow({
                            key: tier if key == "tier" else
                            index if key == "suite_cohort" else
                            ranking_type if key == "ranking_type" else
                            rank if key == "rank" else row.get(key)
                            for key in columns
                        })

    for tier in TIERS:
        tier_report = report["tiers"][tier]
        _write_scatter_svg(
            tier_report["overall"], tier_report["pareto_frontiers"]["speed_psnr"],
            "psnr_db", "PSNR (dB, higher is better)", output / f"{tier}-speed-vs-psnr.svg",
        )
        _write_scatter_svg(
            tier_report["overall"], tier_report["pareto_frontiers"]["speed_lpips"],
            "lpips", "LPIPS (lower is better)", output / f"{tier}-speed-vs-lpips.svg",
        )
        for index, cohort in enumerate(tier_report.get("suite_cohorts", []), 1):
            cohort_report = cohort["report"]
            _write_scatter_svg(
                cohort_report["overall"], cohort_report["pareto_frontiers"]["speed_psnr"],
                "psnr_db", "PSNR (dB, higher is better)", output / f"{tier}-cohort-{index}-speed-vs-psnr.svg",
            )
            _write_scatter_svg(
                cohort_report["overall"], cohort_report["pareto_frontiers"]["speed_lpips"],
                "lpips", "LPIPS (lower is better)", output / f"{tier}-cohort-{index}-speed-vs-lpips.svg",
            )


def _append_markdown_rows(lines: list[str], rows: list[Mapping]) -> None:
    if not rows:
        lines.extend(("No renderer has complete suite coverage in this cohort.", ""))
        return
    lines.extend((
        "| Renderer | Speed index | FPS | FPS 95% CI | Frame ms | PSNR | SSIM | LPIPS | VRAM MB | Efficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for row in rows:
        speed_index = f"{row['speed_index']:.3f}" if row.get("speed_index") is not None else "N/A"
        lines.append(
            f"| {row['renderer']} | {speed_index} | {row['fps']:.2f} | "
            f"{row.get('fps_ci95_low', 0):.2f}-{row.get('fps_ci95_high', 0):.2f} | {row['frame_time_ms']:.3f} | "
            f"{row['psnr_db']:.3f} | {row['ssim']:.4f} | {row['lpips']:.4f} | "
            f"{row['peak_vram_mb']:.0f} | {row['efficiency_score']:.6f} |"
        )
    lines.append("")


def _named_boards(tier_report: Mapping) -> list[tuple[str, list[Mapping]]]:
    return [
        ("overall", tier_report["overall"]),
        ("real_time", tier_report["real_time"]),
        ("quality_psnr", tier_report["quality_rankings"]["psnr"]),
        ("quality_ssim", tier_report["quality_rankings"]["ssim"]),
        ("quality_lpips", tier_report["quality_rankings"]["lpips"]),
        ("efficiency", tier_report["efficiency"]),
        ("memory", tier_report["memory"]),
        ("pareto_combined", tier_report["pareto"]),
    ]


def _append_detailed_boards(lines: list[str], tier_report: Mapping, heading: str) -> None:
    titles = {
        "real_time": "Real-time ranking",
        "quality_psnr": "Quality ranking — PSNR",
        "quality_ssim": "Quality ranking — SSIM",
        "quality_lpips": "Quality ranking — LPIPS",
        "efficiency": "Efficiency ranking",
        "memory": "Memory ranking",
        "pareto_combined": "Combined Pareto ranking",
    }
    titles.update({
        "quality_psnr": "Quality ranking - PSNR",
        "quality_ssim": "Quality ranking - SSIM",
        "quality_lpips": "Quality ranking - LPIPS",
    })
    for name, rows in _named_boards(tier_report):
        if name == "overall":
            continue
        lines.extend((f"{heading} {titles[name]}", ""))
        _append_markdown_rows(lines, rows)


def _write_scatter_svg(rows, frontier_ids, y_metric, y_label, path: Path) -> None:
    """Write a dependency-free Pareto scatter chart, including honest empty state."""
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    frontier = set(frontier_ids)
    if rows:
        xs = [float(row["fps"]) for row in rows]
        ys = [float(row[y_metric]) for row in rows]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = max((x_max - x_min) * .08, max(x_max, 1) * .03)
        y_pad = max((y_max - y_min) * .08, .001)
        x_min, x_max = max(0, x_min - x_pad), x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

    def px(value):
        return left + (float(value) - x_min) / (x_max - x_min) * plot_w

    def py(value):
        return top + (y_max - float(value)) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" fill="#f8fafc" font-family="system-ui" font-size="20">Speed-quality Pareto</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#94a3b8"/>',
        f'<text x="{left + plot_w/2}" y="{height - 20}" text-anchor="middle" fill="#cbd5e1" font-family="system-ui">FPS (higher is better)</text>',
        f'<text transform="translate(24 {top + plot_h/2}) rotate(-90)" text-anchor="middle" fill="#cbd5e1" font-family="system-ui">{html.escape(y_label)}</text>',
    ]
    if not rows:
        parts.append(
            f'<text x="{left + plot_w/2}" y="{top + plot_h/2}" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="18">No complete, comparable results in this evidence tier</text>'
        )
    for row in rows:
        x, y = px(row["fps"]), py(row[y_metric])
        is_frontier = row["competitor_id"] in frontier
        color = "#22c55e" if is_frontier else "#8b5cf6"
        radius = 8 if is_frontier else 6
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')
        parts.append(f'<text x="{x + 10:.2f}" y="{y - 8:.2f}" fill="#e2e8f0" font-family="system-ui" font-size="12">{html.escape(str(row["renderer"]))}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
