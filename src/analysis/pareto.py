"""Pareto-frontier analysis for speed and GT-relative image quality."""
from typing import Iterable, Mapping


QUALITY_METRICS = ("psnr", "ssim", "lpips")


def _dominates(candidate: Mapping, record: Mapping, speed_metric: str) -> bool:
    speed_higher_is_better = speed_metric not in {"latency_ms", "mean_latency_ms", "p99_latency_ms"}
    comparisons = [
        candidate[speed_metric] >= record[speed_metric]
        if speed_higher_is_better else candidate[speed_metric] <= record[speed_metric],
        candidate["psnr"] >= record["psnr"],
        candidate["ssim"] >= record["ssim"],
        candidate["lpips"] <= record["lpips"],
    ]
    strict = [
        candidate[speed_metric] > record[speed_metric]
        if speed_higher_is_better else candidate[speed_metric] < record[speed_metric],
        candidate["psnr"] > record["psnr"],
        candidate["ssim"] > record["ssim"],
        candidate["lpips"] < record["lpips"],
    ]
    return all(comparisons) and any(strict)


def pareto_analysis(records: Iterable[Mapping], speed_metric: str = "fps") -> dict:
    """Return deterministic Pareto membership and dominance evidence."""
    valid = []
    excluded = {}
    for source in records:
        record = dict(source)
        name = record.get("renderer")
        if not name:
            raise ValueError("Every Pareto record needs a renderer name")
        if record.get("benchmark_type") == "synthetic_stress":
            excluded[name] = "synthetic_not_quality_eligible"
        elif record.get(speed_metric) is None:
            excluded[name] = "missing_speed_metric"
        elif any(record.get(metric) is None for metric in QUALITY_METRICS):
            excluded[name] = "missing_quality_metrics"
        else:
            valid.append(record)

    dominated = {}
    for record in valid:
        dominators = sorted(
            candidate["renderer"] for candidate in valid
            if candidate is not record and _dominates(candidate, record, speed_metric)
        )
        if dominators:
            dominated[record["renderer"]] = dominators

    frontier = sorted(
        record["renderer"] for record in valid
        if record["renderer"] not in dominated
    )
    return {
        "schema_version": 1,
        "analysis": "speed_quality_pareto",
        "speed_metric": speed_metric,
        "quality_metrics": {"maximize": ["psnr", "ssim"], "minimize": ["lpips"]},
        "frontier": frontier,
        "dominated": dominated,
        "excluded": dict(sorted(excluded.items())),
    }
