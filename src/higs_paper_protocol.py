"""Validation and expansion of the full-training HiGS paper protocol."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path


class HigsPaperProtocolError(ValueError):
    """Raised when the paper protocol permits an invalid headline experiment."""


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


def _validate_runner_evidence(method_id: str, method: dict) -> None:
    evidence_path = method.get("runner_evidence")
    if not evidence_path:
        raise HigsPaperProtocolError(
            f"ready non-reference method lacks runner evidence: {method_id}"
        )
    path = (_REPO_ROOT / evidence_path).resolve()
    if _REPO_ROOT not in path.parents or not path.is_file():
        raise HigsPaperProtocolError(f"runner evidence is missing: {method_id}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("method") != method_id:
        raise HigsPaperProtocolError(f"runner evidence method mismatch: {method_id}")
    if not (
        evidence.get("run_kind") == "smoke"
        and evidence.get("paper_eligible") is False
        and evidence.get("initialization") == "from_scratch_sfm"
        and evidence.get("clean_process") is True
    ):
        raise HigsPaperProtocolError(f"invalid runner smoke evidence: {method_id}")
    if evidence.get("source", {}).get("commit") != method.get("commit"):
        raise HigsPaperProtocolError(f"runner evidence source mismatch: {method_id}")
    if len(evidence.get("dataset", {}).get("inventory_sha256", "")) != 64:
        raise HigsPaperProtocolError(f"runner evidence lacks dataset hash: {method_id}")
    artifacts = evidence.get("artifacts", [])
    required = ("ckpts/", "stats/train_", "stats/val_")
    if not all(
        any(item.get("path", "").startswith(prefix) and len(item.get("sha256", "")) == 64
            for item in artifacts)
        for prefix in required
    ):
        raise HigsPaperProtocolError(f"runner evidence lacks hashed outputs: {method_id}")


def _selected(value, available):
    return list(available) if value == "all" else list(value)


def build_experiment_plan(protocol: dict) -> list[dict]:
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


def validate_protocol(protocol: dict) -> dict:
    if protocol.get("schema_version") != "1.0" or protocol.get("paper_track") != "higs":
        raise HigsPaperProtocolError("expected schema_version 1.0 for the higs track")
    training = protocol.get("training", {})
    if training.get("initialization") != "from_scratch_sfm":
        raise HigsPaperProtocolError("headline training must use from_scratch_sfm initialization")
    if training.get("iterations", 0) < 30000:
        raise HigsPaperProtocolError("headline training requires at least 30000 iterations")
    seeds = training.get("seeds", [])
    if len(set(seeds)) < 3:
        raise HigsPaperProtocolError("headline training requires at least three unique seeds")

    scenes = protocol.get("scenes", [])
    scene_ids = [scene.get("id") for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)) or len(scene_ids) < 11:
        raise HigsPaperProtocolError("the primary protocol requires 11 unique official scenes")
    families = {scene.get("family") for scene in scenes}
    required_families = {"Mip-NeRF 360", "Tanks and Temples", "Deep Blending"}
    if not required_families.issubset(families):
        raise HigsPaperProtocolError("all three official dataset families are required")

    methods = protocol.get("methods", {})
    primary = next(
        (matrix for matrix in protocol.get("matrices", []) if matrix.get("id") == "primary_full_convergence"),
        None,
    )
    if not primary or primary.get("scenes") != "all" or primary.get("seeds") != "all":
        raise HigsPaperProtocolError("primary_full_convergence must cover all scenes and seeds")
    required_methods = {"original_3dgs", "gsplat", "higs_full", "higs_proposed"}
    if not required_methods.issubset(primary.get("methods", [])):
        raise HigsPaperProtocolError("primary matrix is missing reference or same-backend controls")
    for method_id in primary["methods"]:
        method = methods.get(method_id)
        if not method:
            raise HigsPaperProtocolError(f"unknown method: {method_id}")
        if "proxy" in method.get("implementation", ""):
            raise HigsPaperProtocolError(f"proxy implementation cannot be a headline baseline: {method_id}")
        if method.get("runner_status") == "ready" and not (
            method.get("repository") and method.get("commit")
        ):
            raise HigsPaperProtocolError(f"ready method lacks pinned source identity: {method_id}")
        if (
            method.get("runner_status") == "ready"
            and method.get("role") != "official_reference"
        ):
            _validate_runner_evidence(method_id, method)
        if method.get("patches") and not all(
            len(method.get(key, "")) == 64
            for key in (
                "patch_sha256",
                "source_diff_sha256",
                "source_state_sha256",
                "trainer_sha256",
            )
        ):
            raise HigsPaperProtocolError(
                f"patched method lacks SHA-256 source locks: {method_id}"
            )
        if method.get("runner_status") != "ready" and not method.get("blocking_gate"):
            raise HigsPaperProtocolError(f"blocked method lacks blocking_gate: {method_id}")

    outcomes = set(protocol.get("required_outcomes", []))
    missing_outcomes = sorted(_REQUIRED_OUTCOMES - outcomes)
    if missing_outcomes:
        raise HigsPaperProtocolError(f"missing required outcomes: {missing_outcomes}")

    hardware = protocol.get("hardware", {})
    hardware_classes = {item.get("class") for item in hardware.values()}
    if "consumer" not in hardware_classes or "datacenter" not in hardware_classes:
        raise HigsPaperProtocolError("consumer and data-center hardware classes are required")
    for matrix in protocol["matrices"]:
        unknown = (
            set(matrix["methods"]) - set(methods)
            or set(matrix["hardware"]) - set(hardware)
            or set(_selected(matrix["scenes"], scene_ids)) - set(scene_ids)
        )
        if unknown:
            raise HigsPaperProtocolError(f"{matrix['id']}: unknown matrix member {sorted(unknown)}")

    plan = build_experiment_plan(protocol)
    job_ids = [job["job_id"] for job in plan]
    if len(job_ids) != len(set(job_ids)):
        raise HigsPaperProtocolError("experiment plan contains duplicate job IDs")
    return {
        "status": "protocol_ready",
        "initialization": training["initialization"],
        "iterations": training["iterations"],
        "seed_count": len(set(seeds)),
        "primary_scene_count": len(scene_ids),
        "primary_methods": primary["methods"],
        "hardware_classes": sorted(hardware_classes),
        "planned_jobs": len(plan),
        "executable_jobs": sum(job["executable"] for job in plan),
        "blocked_methods": sorted(
            method_id for method_id, method in methods.items()
            if method["runner_status"] != "ready"
        ),
        "blocked_hardware": sorted(
            hardware_id for hardware_id, item in hardware.items()
            if item["runner_status"] != "ready"
        ),
    }
