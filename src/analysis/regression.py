"""Benchmark regression detection."""
from __future__ import annotations

from typing import Iterable, Mapping


DEFAULT_THRESHOLDS = {
    "fps_drop_pct": 5.0,
    "latency_increase_pct": 5.0,
    "vram_increase_pct": 10.0,
}


def _pct_change(current, baseline):
    if current is None or baseline in (None, 0):
        return None
    return (float(current) - float(baseline)) / float(baseline) * 100.0


def _key(record):
    return (
        record.get("renderer"),
        record.get("benchmark_type"),
        record.get("gaussians"),
        record.get("cohort"),
    )


def compare_regressions(
    baseline_records: Iterable[Mapping],
    candidate_records: Iterable[Mapping],
    thresholds: Mapping[str, float] | None = None,
) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    baseline = {_key(record): dict(record) for record in baseline_records if record.get("renderer")}
    regressions = []
    improvements = []
    unchanged = []

    for candidate in candidate_records:
        candidate = dict(candidate)
        key = _key(candidate)
        base = baseline.get(key)
        if not base:
            continue
        fps_change = _pct_change(candidate.get("fps"), base.get("fps"))
        latency_change = _pct_change(candidate.get("latency_ms"), base.get("latency_ms"))
        vram_change = _pct_change(candidate.get("peak_vram_mb"), base.get("peak_vram_mb"))
        entry = {
            "renderer": candidate.get("renderer"),
            "benchmark_type": candidate.get("benchmark_type"),
            "gaussians": candidate.get("gaussians"),
            "cohort": candidate.get("cohort"),
            "fps_change_pct": None if fps_change is None else round(fps_change, 4),
            "latency_change_pct": None if latency_change is None else round(latency_change, 4),
            "vram_change_pct": None if vram_change is None else round(vram_change, 4),
        }
        failed = (
            fps_change is not None and fps_change < -thresholds["fps_drop_pct"]
            or latency_change is not None and latency_change > thresholds["latency_increase_pct"]
            or vram_change is not None and vram_change > thresholds["vram_increase_pct"]
        )
        if failed:
            regressions.append({"status": "regression", **entry})
        elif fps_change is not None and fps_change > thresholds["fps_drop_pct"]:
            improvements.append({"status": "improvement", **entry})
        else:
            unchanged.append({"status": "unchanged", **entry})

    return {
        "schema_version": 1,
        "status": "regression" if regressions else "ok",
        "thresholds": thresholds,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
    }

