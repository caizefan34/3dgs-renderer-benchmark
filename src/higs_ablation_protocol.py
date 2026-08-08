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

_REQUIRED_CONFIRMATORY_METHODS = {
    "gsplat",
    "higs_full",
    "higs_current",
    "higs_switch_12k",
    "higs_switch_21k",
}

# Confirmatory matrices are fail-closed: each frozen id pins its exact method
# set, matched controls, and frozen candidates. New confirmatory matrices must
# be added here (with tests) before they can appear in a protocol.
_CONFIRMATORY_MATRIX_SPECS = {
    "confirmatory_formal_30k": {
        "methods": frozenset(_REQUIRED_CONFIRMATORY_METHODS),
        "matched_controls": frozenset({"gsplat", "higs_full", "higs_current"}),
        "frozen_candidates": ("higs_switch_12k", "higs_switch_21k"),
    },
    "confirmatory_higs_sched_11s3": {
        "methods": frozenset({
            "gsplat",
            "gsplat_27k",
            "gsplat_27k_preload_accum8",
            "higs_sched_27k",
        }),
        "matched_controls": frozenset({
            "gsplat",
            "gsplat_27k",
            "gsplat_27k_preload_accum8",
        }),
        "frozen_candidates": ("higs_sched_27k",),
    },
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
                "iterations": (method_spec.get("algorithm") or {}).get("max_steps") or training["iterations"],
                "executable": (
                    method_spec["runner_status"] == "ready"
                    and hardware_spec["runner_status"] == "ready"
                ),
            })
    return jobs


def _validate_confirmatory_matrix(
    matrix: dict, methods: dict, scene_ids: list[str], training_seeds: list
) -> tuple:
    """Fail-closed checks for a frozen confirmatory matrix; returns its frozen candidates."""
    spec = _CONFIRMATORY_MATRIX_SPECS.get(matrix["id"])
    if spec is None:
        raise HigsAblationProtocolError(
            f"{matrix['id']}: unknown confirmatory matrix id; add it to "
            "_CONFIRMATORY_MATRIX_SPECS with tests"
        )
    if set(matrix["methods"]) != spec["methods"]:
        raise HigsAblationProtocolError(
            f"{matrix['id']}: confirmatory methods must be exactly "
            f"{sorted(spec['methods'])}"
        )
    if _selected(matrix["scenes"], scene_ids) != scene_ids:
        raise HigsAblationProtocolError(
            f"{matrix['id']}: confirmatory matrix must cover all 11 scenes"
        )
    seeds = _selected(matrix["seeds"], training_seeds)
    if len(set(seeds)) < 3:
        raise HigsAblationProtocolError(
            f"{matrix['id']}: confirmatory matrix requires >= 3 unique seeds"
        )
    controls = matrix.get("matched_controls", [])
    if set(controls) != spec["matched_controls"]:
        raise HigsAblationProtocolError(
            f"{matrix['id']}: matched_controls must be {sorted(spec['matched_controls'])}"
        )
    return spec["frozen_candidates"]


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
    if len(scene_ids) != len(set(scene_ids)) or len(scene_ids) < 11:
        raise HigsAblationProtocolError(
            "the ablation protocol requires 11 unique official scenes"
        )
    families = {scene.get("family") for scene in scenes}
    required_families = {"Mip-NeRF 360", "Tanks and Temples", "Deep Blending"}
    if not required_families.issubset(families):
        raise HigsAblationProtocolError("all three official dataset families are required")

    methods = protocol.get("methods", {})
    missing = _REQUIRED_ABLATION_METHODS - set(methods)
    if missing:
        raise HigsAblationProtocolError(
            f"ablation protocol is missing required methods: {sorted(missing)}"
        )
    for method_id, method in methods.items():
        override = (method.get("algorithm") or {}).get("max_steps")
        if override is not None and (
            not isinstance(override, int) or not (1 <= override <= 30_000)
        ):
            raise HigsAblationProtocolError(
                f"method {method_id}: algorithm.max_steps must be an integer in [1, 30000]"
            )
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
        if matrix.get("phase") == "confirmatory":
            _validate_confirmatory_matrix(matrix, methods, scene_ids, seeds)

    frozen = protocol.get("frozen_candidates", [])
    required_frozen = set()
    for matrix in protocol.get("matrices", []):
        if matrix.get("phase") == "confirmatory":
            required_frozen.update(_validate_confirmatory_matrix(matrix, methods, scene_ids, seeds))
    if not required_frozen:
        # Exploration-only protocols keep the legacy frozen-candidate default.
        required_frozen = set(("higs_switch_12k", "higs_switch_21k"))
    if set(frozen) != required_frozen:
        raise HigsAblationProtocolError(
            f"confirmatory phase requires frozen_candidates == {sorted(required_frozen)}"
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
        "scene_count": len(scene_ids),
        "planned_jobs": len(plan),
        "executable_jobs": sum(job["executable"] for job in plan),
        "confirmatory_jobs": sum(
            job["matrix"] in _CONFIRMATORY_MATRIX_SPECS for job in plan
        ),
        "blocked_methods": sorted(
            method_id for method_id, method in methods.items()
            if method["runner_status"] != "ready"
        ),
    }
