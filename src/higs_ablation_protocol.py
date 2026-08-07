"""Validation and expansion of the independent HiGS ablation protocol."""
from __future__ import annotations

from itertools import product
from pathlib import Path


class HigsAblationProtocolError(ValueError):
    """Raised when the ablation protocol permits an invalid pilot experiment."""


_REQUIRED_OUTCOMES = {
    "wall_time_seconds",
    "time_to_quality_seconds",
    "psnr_db",
    "ssim",
    "lpips",
    "peak_gpu_memory_mib",
    "final_gaussian_count",
}

_REPO_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_ABLATION_METHODS = {
    "higs_full",
    "higs_visible_only",
    "higs_progressive_only",
    "higs_current",
}


def _selected(value, available):
    return list(available) if value == "all" else list(value)


def build_ablation_experiment_plan(protocol: dict) -> list[dict]:
    """Expand the ablation matrices into auditable per-GPU jobs."""
    training = protocol["training"]
    scene_ids = [scene["id"] for scene in protocol["scenes"]]
    jobs = []
    for matrix in protocol["matrices"]:
        scenes = _selected(matrix["scenes"], scene_ids)
        seeds = _selected(matrix["seeds"], training["seeds"])
        for method, scene, hardware, seed in product(
            matrix["methods"], scenes, matrix["hardware"], seeds
        ):
            method_spec = protocol["methods"][method]
            hardware_spec = protocol["hardware"][hardware]
            jobs.append({
                "job_id": f"{matrix['id']}--{method}--{scene.replace('/', '-')}--{hardware}--s{seed}",
                "matrix": matrix["id"],
                "method": method,
                "scene": scene,
                "hardware": hardware,
                "seed": seed,
                "initialization": training["initialization"],
                "iterations": training["iterations"],
                "executable": (
                    method_spec["runner_status"] == "ready"
                    and hardware_spec["runner_status"] == "ready"
                ),
            })
    return jobs


def validate_ablation_protocol(protocol: dict) -> dict:
    """Fail-closed validation of the independent ablation/sensitivity protocol."""
    if protocol.get("schema_version") != "1.0" or protocol.get("paper_track") != "higs_ablation":
        raise HigsAblationProtocolError(
            "expected schema_version 1.0 for the higs_ablation track"
        )
    training = protocol.get("training", {})
    if training.get("initialization") != "from_scratch_sfm":
        raise HigsAblationProtocolError(
            "ablation training must use from_scratch_sfm initialization"
        )
    if training.get("iterations", 0) < 30000:
        raise HigsAblationProtocolError("ablation training requires at least 30000 iterations")
    seeds = training.get("seeds", [])
    if not seeds:
        raise HigsAblationProtocolError("ablation training requires at least one seed")

    scenes = protocol.get("scenes", [])
    scene_ids = [scene.get("id") for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)) or len(scene_ids) < 5:
        raise HigsAblationProtocolError("the ablation protocol requires 5 unique pilot scenes")

    methods = protocol.get("methods", {})
    missing = _REQUIRED_ABLATION_METHODS - set(methods)
    if missing:
        raise HigsAblationProtocolError(
            f"ablation protocol is missing required methods: {sorted(missing)}"
        )
    for method_id, method in methods.items():
        if method.get("runner_status") == "ready" and not (
            method.get("repository") and method.get("commit")
        ):
            raise HigsAblationProtocolError(
                f"ready method lacks pinned source identity: {method_id}"
            )
        if method.get("patches") and not all(
            len(method.get(key, "")) == 64
            for key in ("patch_sha256", "source_diff_sha256", "source_state_sha256", "trainer_sha256")
        ):
            raise HigsAblationProtocolError(
                f"patched method lacks SHA-256 source locks: {method_id}"
            )
        if method.get("runner_status") != "ready" and not method.get("blocking_gate"):
            raise HigsAblationProtocolError(f"blocked method lacks blocking_gate: {method_id}")

    outcomes = set(protocol.get("required_outcomes", []))
    missing_outcomes = sorted(_REQUIRED_OUTCOMES - outcomes)
    if missing_outcomes:
        raise HigsAblationProtocolError(f"missing required outcomes: {missing_outcomes}")

    hardware = protocol.get("hardware", {})
    if "a100" not in hardware:
        raise HigsAblationProtocolError("ablation protocol requires the a100 cohort")
    for matrix in protocol.get("matrices", []):
        unknown = (
            set(matrix["methods"]) - set(methods)
            or set(matrix["hardware"]) - set(hardware)
            or set(_selected(matrix["scenes"], scene_ids)) - set(scene_ids)
        )
        if unknown:
            raise HigsAblationProtocolError(
                f"{matrix['id']}: unknown matrix member {sorted(unknown)}"
            )

    plan = build_ablation_experiment_plan(protocol)
    job_ids = [job["job_id"] for job in plan]
    if len(job_ids) != len(set(job_ids)):
        raise HigsAblationProtocolError("experiment plan contains duplicate job IDs")

    return {
        "status": "ablation_protocol_ready",
        "initialization": training["initialization"],
        "iterations": training["iterations"],
        "seed_count": len(set(seeds)),
        "pilot_scene_count": len(scene_ids),
        "planned_jobs": len(plan),
        "executable_jobs": sum(job["executable"] for job in plan),
        "blocked_methods": sorted(
            method_id for method_id, method in methods.items()
            if method["runner_status"] != "ready"
        ),
    }
