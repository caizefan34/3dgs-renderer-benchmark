#!/usr/bin/env python
"""Merge one local speed run and quality run into a strict Tier A record."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_matrix import validate_result  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _speed_row(document, renderer):
    rows = document.get("results", {})
    if isinstance(rows, dict):
        row = rows.get(renderer)
        if row is None:
            row = next((value for value in rows.values() if value.get("renderer") == renderer), None)
    else:
        row = next((value for value in rows if value.get("renderer") == renderer), None)
    if row is None:
        raise ValueError(f"speed artifact has no row for {renderer}")
    return row


def _quality_row(document, renderer):
    row = next((value for value in document.get("results", []) if value.get("renderer") == renderer), None)
    if row is None:
        raise ValueError(f"quality artifact has no row for {renderer}")
    return row


def _expected_quality_images(camera_path: Path, ground_truth_dir: Path) -> list[str]:
    document = _load(camera_path)
    cameras = document if isinstance(document, list) else document.get("cameras", [])
    image_index = {
        path.stem: path.name for path in ground_truth_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    expected = []
    for camera in cameras:
        name = camera.get("image_name") or camera.get("img_name")
        stem = Path(name).stem if name else None
        if stem not in image_index:
            raise ValueError(f"canonical camera {name!r} has no GT image")
        expected.append(image_index[stem])
    return expected


def _verify_render_outputs(quality_document: dict, quality_frames: list[dict]) -> list[dict]:
    root_value = quality_document.get("render_outputs", {}).get("root")
    if not root_value:
        raise ValueError("quality artifact lacks lossless render-output root")
    root = Path(root_value).resolve()
    verified = []
    for frame in quality_frames:
        record = frame.get("render_output")
        if not record or not all(
            key in record for key in (
                "path", "sha256", "format", "source_tensor_dtype",
                "source_tensor_shape", "export_encoding",
            )
        ):
            raise ValueError("quality frame lacks lossless render-output evidence")
        if record["format"] != "png" or not record["path"].lower().endswith(".png"):
            raise ValueError("render output must be a PNG")
        path = (root / record["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("render-output path escapes its evidence root") from exc
        if not path.is_file():
            raise ValueError(f"render output is missing: {record['path']}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"render output hash mismatch: {record['path']}")
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.load()
                source_shape = record["source_tensor_shape"]
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError(f"render output is not RGB PNG: {record['path']}")
                if len(source_shape) != 3 or source_shape[-1] != 3:
                    raise ValueError("render-output source tensor shape is not HWC RGB")
                if image.size != (source_shape[1], source_shape[0]):
                    raise ValueError(f"render-output dimensions mismatch: {record['path']}")
        except OSError as exc:
            raise ValueError(f"render output is not a valid PNG: {record['path']}") from exc
        verified.append({
            "frame": frame.get("frame"),
            "image": frame.get("image"),
            **record,
        })
    return verified


def collect(run_dir: Path, renderer_id: str, case_id: str) -> dict:
    suite = _load(ROOT / "benchmark" / "suite.json")
    protocol = _load(ROOT / "benchmark" / "protocol.json")
    actual_protocol_sha = _sha256(ROOT / "benchmark" / "protocol.json")
    if actual_protocol_sha != suite["protocol_sha256"]:
        raise ValueError("suite protocol hash does not match benchmark/protocol.json")
    registry = _load(ROOT / "benchmark" / "renderers.json")
    case = next(row for row in suite["cases"] if row["case_id"] == case_id)
    renderer = next(row for row in registry["renderers"] if renderer_id in row["adapter_ids"])
    if renderer["execution_status"] != "automatic_ready":
        raise ValueError(f"{renderer_id} is not eligible for automatic Tier A collection")
    if renderer_id in renderer.get("primary_excluded_adapter_ids", []):
        raise ValueError(f"{renderer_id} is excluded from the primary recommendation track")

    speed_path = run_dir / renderer_id / "speed" / "benchmark_results.json"
    nvml_samples_path = run_dir / renderer_id / "speed" / "nvml_samples.json"
    quality_path = run_dir / renderer_id / "quality" / "quality_gt.json"
    speed_doc, quality_doc = _load(speed_path), _load(quality_path)
    nvml_samples_doc = _load(nvml_samples_path)
    nvml_samples = nvml_samples_doc.get("renderers", {}).get(renderer_id, [])
    if not nvml_samples:
        raise ValueError("speed artifact lacks raw NVML process-memory samples")
    for sample in nvml_samples:
        if not all(key in sample for key in ("relative_ms", "timestamp_utc", "pid", "used_gpu_memory_mib")):
            raise ValueError("raw NVML sample is incomplete")
    if speed_doc.get("status") != "complete":
        raise ValueError("speed artifact is not complete")
    speed, quality_result = _speed_row(speed_doc, renderer_id), _quality_row(quality_doc, renderer_id)
    observed_commit = speed.get("renderer_commit_hash")
    if not observed_commit:
        raise ValueError(
            "renderer commit is not discoverable; install from a pinned VCS checkout or PEP 610 direct URL"
        )
    if observed_commit != renderer["source_commit"]:
        raise ValueError(
            f"renderer commit mismatch: expected {renderer['source_commit']}, got {observed_commit}"
        )
    quality_commit = quality_result.get("metadata", {}).get("commit_hash")
    if quality_commit != observed_commit:
        raise ValueError("speed and quality renderer commits do not match")
    quality = quality_result["quality"]
    wall_samples = speed.get("wall_frame_times_ms") or speed.get("frame_times_ms")
    if not wall_samples:
        raise ValueError("speed artifact lacks raw frame-time samples")
    wall_samples = [float(value) for value in wall_samples]
    repeat_wall_samples = speed.get("repeat_wall_frame_times_ms")
    repeat_gpu_samples = speed.get("repeat_frame_times_ms")
    if not repeat_wall_samples or not repeat_gpu_samples:
        raise ValueError("speed artifact lacks repeat-level raw samples")
    quality_frames = quality_result.get("frames", [])
    if not quality_frames:
        raise ValueError("quality artifact lacks per-view samples")
    render_outputs = _verify_render_outputs(quality_doc, quality_frames)
    if speed.get("nvml_peak_vram_mb") is None:
        raise ValueError("Tier A requires NVML process peak memory; install nvidia-ml-py")

    raw = {
        "schema_version": 1,
        "renderer_id": renderer_id,
        "case_id": case_id,
        "wall_frame_times_ms": wall_samples,
        "gpu_frame_times_ms": speed.get("frame_times_ms", []),
        "repeat_wall_frame_times_ms": repeat_wall_samples,
        "repeat_gpu_frame_times_ms": repeat_gpu_samples,
        "quality_frames": quality_frames,
        "render_output_root": quality_doc["render_outputs"]["root"],
        "render_outputs": render_outputs,
        "nvml_process_memory_samples": nvml_samples,
        "nvml_sampling_interval_ms": nvml_samples_doc.get("sampling_interval_ms"),
        "source_speed_sha256": _sha256(speed_path),
        "source_quality_sha256": _sha256(quality_path),
        "source_nvml_samples_sha256": _sha256(nvml_samples_path),
    }
    raw_path = run_dir / "raw_samples.json"
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw_sha = _sha256(raw_path)

    environment = speed_doc.get("environment", {})
    hardware = speed.get("hardware_metadata") or {}
    dataset_manifest = ROOT / case["dataset_manifest"]
    dataset_spec = _load(dataset_manifest)
    preparation_path = ROOT / case["preparation_path"]
    preparation = _load(preparation_path)
    scene_path = ROOT / case["scene_path"]
    camera_path = ROOT / case["camera_path"]
    ground_truth_dir = ROOT / case["ground_truth_path"]
    scene_spec = dataset_spec.get("scenes", {}).get(case["scene_id"])
    if not scene_spec:
        raise ValueError("dataset manifest does not pin this scene")
    inventory_path = (
        ROOT / "datasets" / "raw" / case["dataset_id"] /
        case["scene_id"] / "dataset_inventory.json"
    )
    inventory = _load(inventory_path)
    source_verification = inventory.get("source_verification", {})
    if source_verification.get("size_bytes") != scene_spec["source"].get("size_bytes"):
        raise ValueError("processed case does not match the pinned dataset source size")
    for key, expected in scene_spec["source"].get("checksums", {}).items():
        if source_verification.get(key) != expected:
            raise ValueError(f"processed case does not match the pinned dataset {key}")
    dataset_sha256 = inventory.get("source_archive_sha256")
    if not dataset_sha256 or preparation.get("source_archive_sha256") != dataset_sha256:
        raise ValueError("processed case does not match the verified local source archive")
    if preparation.get("checkpoint_sha256") != _sha256(scene_path):
        raise ValueError("processed checkpoint hash mismatch")
    if preparation.get("camera_trajectory_sha256") != _sha256(camera_path):
        raise ValueError("processed camera hash mismatch")
    canonical = case["canonical_assets"]
    if canonical.get("status") != "pinned":
        raise ValueError("suite canonical assets are not published")
    for key, actual in (
        ("checkpoint_sha256", preparation.get("checkpoint_sha256")),
        ("camera_sha256", preparation.get("camera_trajectory_sha256")),
        ("ground_truth_manifest_sha256", preparation.get("ground_truth_file_manifest_sha256")),
    ):
        if canonical.get(key) != actual:
            raise ValueError(f"canonical asset mismatch: {key}")
    width, height = protocol["resolution"]
    expected_scene_sha = _sha256(scene_path)
    expected_camera_sha = _sha256(camera_path)
    speed_scene = speed_doc.get("scene", {})
    speed_protocol = speed_doc.get("protocol", {})
    if speed_scene.get("model_sha256") != expected_scene_sha:
        raise ValueError("speed artifact checkpoint hash mismatch")
    if speed_scene.get("camera_sha256") != expected_camera_sha:
        raise ValueError("speed artifact camera hash mismatch")
    if speed_protocol.get("resolution") != [width, height]:
        raise ValueError("speed artifact resolution mismatch")
    for key, expected in (
        ("warmup_frames", protocol["timing"]["warmup_frames"]),
        ("measured_frames_per_repeat", protocol["timing"]["measured_frames_per_repeat"]),
        ("repeats", protocol["timing"]["repeats"]),
    ):
        if speed_protocol.get(key) != expected or speed.get(key) != expected:
            raise ValueError(f"speed artifact {key} mismatch")
    if speed.get("image_width") != width or speed.get("image_height") != height:
        raise ValueError("speed row resolution mismatch")
    expected_sample_count = protocol["timing"]["measured_frames_per_repeat"] * protocol["timing"]["repeats"]
    if len(wall_samples) != expected_sample_count:
        raise ValueError("speed raw sample count mismatch")
    expected_repeat_count = protocol["timing"]["repeats"]
    expected_frames = protocol["timing"]["measured_frames_per_repeat"]
    if len(repeat_wall_samples) != expected_repeat_count or any(len(repeat) != expected_frames for repeat in repeat_wall_samples):
        raise ValueError("wall-clock repeat structure mismatch")
    if len(repeat_gpu_samples) != expected_repeat_count or any(len(repeat) != expected_frames for repeat in repeat_gpu_samples):
        raise ValueError("GPU repeat structure mismatch")

    quality_reference = quality_doc.get("reference", {})
    if quality_doc.get("scene_sha256") != expected_scene_sha:
        raise ValueError("quality artifact checkpoint hash mismatch")
    if quality_reference.get("camera_manifest_sha256") != expected_camera_sha:
        raise ValueError("quality artifact camera hash mismatch")
    if quality_reference.get("resolution") != [width, height]:
        raise ValueError("quality artifact resolution mismatch")
    if preparation.get("ground_truth_file_manifest_sha256") != quality_reference.get("ground_truth_manifest_sha256"):
        raise ValueError("staged ground-truth manifest hash mismatch")
    expected_images = _expected_quality_images(camera_path, ground_truth_dir)
    actual_images = [frame.get("image") for frame in quality_frames]
    if actual_images != expected_images:
        raise ValueError("quality views do not exactly match the ordered camera manifest")
    if quality.get("num_views") != len(expected_images):
        raise ValueError("quality view count mismatch")
    measured_at = datetime.now(timezone.utc).isoformat()
    fps = len(wall_samples) / (sum(wall_samples) / 1000.0)
    repeat_fps = [len(repeat) / (sum(repeat) / 1000.0) for repeat in repeat_wall_samples]
    fps_sem = statistics.stdev(repeat_fps) / math.sqrt(len(repeat_fps)) if len(repeat_fps) > 1 else 0.0
    t_critical_95 = 2.776 if len(repeat_fps) == 5 else 1.96
    fps_ci_low = max(0.0, statistics.mean(repeat_fps) - t_critical_95 * fps_sem)
    fps_ci_high = statistics.mean(repeat_fps) + t_critical_95 * fps_sem
    ordered = sorted(wall_samples)

    def percentile(fraction):
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    identity = "|".join((renderer_id, case_id, measured_at, _git_commit()))
    result = {
        "schema_version": "2.0",
        "result_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "evidence_tier": "measured",
        "status": "complete",
        "renderer": {
            "id": renderer_id,
            "config_id": renderer_id,
            "name": f"{renderer['name']} ({renderer_id})",
            "version": speed.get("renderer_version") or "unknown",
            "source_uri": renderer["source_uri"],
            "source_commit": renderer["source_commit"],
            "build_command": renderer["build"],
            "runtime_command": renderer["runtime"],
            "api": renderer["api"],
            "backend": renderer["backend"],
            "platforms": renderer["platforms"],
            "features": renderer["features"],
        },
        "benchmark": {
            "suite_id": suite["suite_id"], "suite_version": suite["version"],
            "track_id": suite["primary_track"], "case_id": case_id,
            "dataset_id": case["dataset_id"], "dataset_sha256": dataset_sha256,
            "scene_id": case["scene_id"], "scene_tier": case["workload_tier"],
            "checkpoint_sha256": expected_scene_sha,
            "gaussian_count": int(speed.get("num_gaussians", 0)),
            "sh_degree": suite["common_representation"]["sh_degree"],
            "camera_trajectory_id": f"{case_id}-eval",
            "camera_trajectory_sha256": expected_camera_sha,
            "quality_reference_sha256": quality_doc["reference"]["ground_truth_manifest_sha256"],
            "resolution": {"width": width, "height": height},
            "color_space": protocol["color_space"], "background": protocol["background"],
            "protocol_id": protocol["protocol_id"], "protocol_sha256": suite["protocol_sha256"],
        },
        "environment": {
            "hardware_profile_id": hashlib.sha256(json.dumps({
                "gpu": environment.get("gpu"), "driver": environment.get("driver"),
                "cpu": environment.get("cpu"), "os": environment.get("os"),
            }, sort_keys=True).encode()).hexdigest()[:16],
            "gpu": environment.get("gpu") or speed.get("gpu") or hardware.get("gpu_name") or "unknown",
            "gpu_uuid": environment.get("gpu_uuid"),
            "gpu_vram_mb": environment.get("gpu_vram_mb") or hardware.get("total_vram_mb") or 0,
            "cpu": environment.get("cpu") or platform.processor() or platform.machine(),
            "ram_mb": environment.get("ram_mb") or 0,
            "os": environment.get("os") or platform.platform(),
            "driver": environment.get("driver") or speed.get("driver_version") or "unknown",
            "cuda": environment.get("cuda_runtime") or speed.get("cuda_version") or "unknown",
            "python": environment.get("python") or platform.python_version(),
            "pytorch": environment.get("pytorch") or "unknown",
            "benchmark_commit": speed.get("benchmark_commit_hash") or _git_commit(),
            "clock_policy": "locked" if environment.get("gpu_clocks_locked") else "default",
            "power_limit_w": environment.get("power_limit_w"),
        },
        "metrics": {
            "performance": {
                "fps": fps,
                "fps_ci95_low": fps_ci_low,
                "fps_ci95_high": fps_ci_high,
                "frame_time_ms": statistics.mean(wall_samples),
                "p95_frame_time_ms": percentile(.95), "p99_frame_time_ms": percentile(.99),
                "peak_vram_mb": speed["nvml_peak_vram_mb"],
                "startup_time_ms": speed["startup_time_ms"],
                "renderer_init_time_ms": speed["renderer_init_time_ms"],
                "scene_load_time_ms": speed["scene_load_time_ms"],
                "renderer_prepare_time_ms": speed["renderer_prepare_time_ms"],
                "time_to_first_frame_ms": speed["time_to_first_frame_ms"],
            },
            "quality": {
                "psnr_db": quality["mean_psnr_db"], "ssim": quality["mean_ssim"],
                "lpips": quality["mean_lpips"],
            },
            "raw_samples": {"uri": str(raw_path.relative_to(ROOT)).replace("\\", "/"), "sha256": raw_sha},
        },
        "provenance": {
            "source_type": "repository_run", "source_uri": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
            "measured_at": measured_at, "raw_samples_uri": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "raw_samples_sha256": raw_sha, "deviations": [],
        },
    }
    validate_result(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run_dir.resolve(), args.renderer, args.case_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
