"""Deterministic recommendation rules for evaluation-framework outputs."""
from typing import Iterable, Mapping, Optional


FAILED_QUALITY_STATUSES = {
    "failed",
    "not_measured",
    "not_measured_no_ground_truth",
    "quality_regression",
}


def _quality_eligible(record):
    return (
        record.get("benchmark_type") != "synthetic_stress"
        and record.get("quality_status") not in FAILED_QUALITY_STATUSES
    )


def _select(records, key, *, reverse=True, candidates=None, quality_eligible=False):
    eligible = [
        record for record in records
        if record.get(key) is not None
        and (candidates is None or record["renderer"] in candidates)
        and (not quality_eligible or _quality_eligible(record))
    ]
    if not eligible:
        return None
    # Renderer id is the stable final tie breaker.
    ordered = sorted(eligible, key=lambda record: record["renderer"])
    value = (max if reverse else min)(record[key] for record in ordered)
    winner = next(record for record in ordered if record[key] == value)
    return {"renderer": winner["renderer"], "metric": key, "value": value}


def build_recommendations(
    records: Iterable[Mapping],
    pareto_renderers: Optional[Iterable[str]] = None,
) -> dict:
    """Choose one renderer per documented category without hidden weights."""
    records = [dict(record) for record in records]
    for record in records:
        if not record.get("renderer"):
            raise ValueError("Every recommendation record needs a renderer name")
    pareto_set = set(pareto_renderers) if pareto_renderers is not None else None
    return {
        "schema_version": 1,
        "rules_version": "deterministic_v1",
        "recommendations": {
            "best_absolute_speed": _select(records, "fps"),
            "best_quality_preserving": _select(
                records, "quality_factor", quality_eligible=True
            ),
            "best_balanced": _select(
                records, "effective_fps", quality_eligible=True
            ),
            "best_memory_efficiency": _select(records, "peak_vram_mb", reverse=False),
            "best_low_latency": _select(records, "p99_latency_ms", reverse=False),
            "best_pareto_candidate": _select(
                records, "effective_fps", candidates=pareto_set,
                quality_eligible=True,
            ),
        },
    }
