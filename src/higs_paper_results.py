"""Fail-closed validation for full-training HiGS paper results."""
from __future__ import annotations

import math

from higs_paper_protocol import build_experiment_plan, validate_protocol


class HigsPaperResultError(ValueError):
    """Raised when a result cannot enter the paper matrix."""


def _finite(value, label: str, *, minimum=None, maximum=None):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HigsPaperResultError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise HigsPaperResultError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise HigsPaperResultError(f"{label} must be <= {maximum}")


def validate_result(result: dict, protocol: dict) -> None:
    validate_protocol(protocol)
    jobs = {job["job_id"]: job for job in build_experiment_plan(protocol)}
    job_id = result.get("job_id")
    if result.get("schema_version") != "1.0" or job_id not in jobs:
        raise HigsPaperResultError(f"unknown or malformed job_id: {job_id!r}")
    job = jobs[job_id]
    for field in ("method", "scene", "hardware", "seed"):
        if result.get(field) != job[field]:
            raise HigsPaperResultError(f"{job_id}: {field} does not match the plan")

    status = result.get("status")
    if status == "failed":
        if not result.get("failure", {}).get("reason"):
            raise HigsPaperResultError(f"{job_id}: failed result requires a reason")
        return
    if status != "complete":
        raise HigsPaperResultError(f"{job_id}: status must be complete or failed")

    training = result.get("training", {})
    if training.get("initialization") != job["initialization"]:
        raise HigsPaperResultError(f"{job_id}: initialization must match from-scratch protocol")
    if training.get("iterations") != job["iterations"]:
        raise HigsPaperResultError(f"{job_id}: iteration budget does not match the plan")

    performance = result.get("performance", {})
    _finite(performance.get("wall_time_seconds"), "wall_time_seconds", minimum=0.0)
    _finite(performance.get("time_to_quality_seconds"), "time_to_quality_seconds", minimum=0.0)
    if performance["time_to_quality_seconds"] > performance["wall_time_seconds"]:
        raise HigsPaperResultError(f"{job_id}: time-to-quality exceeds total wall time")

    quality = result.get("quality", {})
    _finite(quality.get("psnr_db"), "psnr_db")
    _finite(quality.get("ssim"), "ssim", minimum=0.0, maximum=1.0)
    _finite(quality.get("lpips"), "lpips", minimum=0.0)
    resources = result.get("resources", {})
    _finite(resources.get("peak_gpu_memory_mib"), "peak_gpu_memory_mib", minimum=0.0)
    _finite(resources.get("energy_joules"), "energy_joules", minimum=0.0)
    _finite(resources.get("final_gaussian_count"), "final_gaussian_count", minimum=1)

    curve = result.get("quality_curve", [])
    if len(curve) < 2:
        raise HigsPaperResultError(f"{job_id}: quality_curve requires at least two checkpoints")
    previous_iteration = -1
    previous_wall = -1.0
    for index, point in enumerate(curve):
        iteration = point.get("iteration")
        wall = point.get("wall_time_seconds")
        _finite(iteration, f"quality_curve[{index}].iteration", minimum=0)
        _finite(wall, f"quality_curve[{index}].wall_time_seconds", minimum=0.0)
        if iteration <= previous_iteration or wall <= previous_wall:
            raise HigsPaperResultError(f"{job_id}: quality curve must be strictly ordered")
        previous_iteration, previous_wall = iteration, wall
        _finite(point.get("psnr_db"), f"quality_curve[{index}].psnr_db")
        _finite(point.get("ssim"), f"quality_curve[{index}].ssim", minimum=0.0, maximum=1.0)
        _finite(point.get("lpips"), f"quality_curve[{index}].lpips", minimum=0.0)
    if curve[-1]["iteration"] != job["iterations"]:
        raise HigsPaperResultError(f"{job_id}: quality curve final iteration must equal the budget")

    digest = result.get("artifact", {}).get("sha256", "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        raise HigsPaperResultError(f"{job_id}: artifact requires a SHA-256 digest")
    provenance = result.get("provenance", {})
    expected_boundary = protocol["training"]["timing_boundary"]
    if provenance.get("timing_boundary") != expected_boundary or provenance.get("clean_process") is not True:
        raise HigsPaperResultError(f"{job_id}: invalid timing boundary or process provenance")


def validate_result_set(results: list[dict], protocol: dict, require_complete: bool = False) -> dict:
    plan = build_experiment_plan(protocol)
    planned_ids = {job["job_id"] for job in plan}
    seen = set()
    complete = failed = 0
    for result in results:
        job_id = result.get("job_id")
        if job_id in seen:
            raise HigsPaperResultError(f"duplicate job_id: {job_id}")
        seen.add(job_id)
        validate_result(result, protocol)
        complete += result["status"] == "complete"
        failed += result["status"] == "failed"
    missing = len(planned_ids - seen)
    if require_complete and (failed or missing):
        raise HigsPaperResultError(
            f"paper matrix incomplete: complete={complete}, failed={failed}, missing={missing}"
        )
    return {"planned": len(plan), "complete": complete, "failed": failed, "missing": missing}
