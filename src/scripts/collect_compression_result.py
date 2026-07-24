#!/usr/bin/env python
"""Collect one EPIC-05 decoded-PLY run into the compression track."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from schema_validation import validate_schema  # noqa: E402
from scripts.collect_matrix_result import (  # noqa: E402
    _expected_quality_images,
    _load,
    _quality_row,
    _sha256,
    _speed_row,
    _verify_render_outputs,
)


LIMITS = {"max_psnr_drop_db": 0.2, "max_ssim_drop": 0.002, "max_lpips_increase": 0.005}


def quality_gate(reference: dict, candidate: dict, visual_audit: str) -> tuple[dict, dict]:
    delta = {
        "psnr_db": candidate["psnr_db"] - reference["psnr_db"],
        "ssim": candidate["ssim"] - reference["ssim"],
        "lpips": candidate["lpips"] - reference["lpips"],
    }
    numeric_pass = (
        delta["psnr_db"] > -LIMITS["max_psnr_drop_db"]
        and delta["ssim"] > -LIMITS["max_ssim_drop"]
        and delta["lpips"] < LIMITS["max_lpips_increase"]
    )
    gate = {
        **LIMITS,
        "numeric_pass": numeric_pass,
        "visual_audit": visual_audit,
        "overall_pass": numeric_pass and visual_audit in {"pass", "not_applicable"},
    }
    return delta, gate


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _environment(speed_document: dict, speed: dict) -> dict:
    environment = speed_document.get("environment", {})
    identity = {
        key: environment.get(key)
        for key in ("gpu", "driver", "cuda_runtime", "pytorch", "os")
    }
    return {
        **environment,
        "hardware_profile_id": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:16],
        "benchmark_commit": speed.get("benchmark_commit_hash") or _git_commit(),
    }


def _performance(speed: dict) -> dict:
    wall = [float(value) for value in speed.get("wall_frame_times_ms", [])]
    repeat_wall = speed.get("repeat_wall_frame_times_ms", [])
    if not wall or not repeat_wall:
        raise ValueError("compression speed run lacks wall-clock raw samples")
    ordered = sorted(wall)
    repeat_fps = [len(row) / (sum(row) / 1000.0) for row in repeat_wall]
    sem = statistics.stdev(repeat_fps) / math.sqrt(len(repeat_fps)) if len(repeat_fps) > 1 else 0.0
    critical = 2.776 if len(repeat_fps) == 5 else 1.96
    mean_repeat_fps = statistics.mean(repeat_fps)

    def percentile(fraction):
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "fps": len(wall) / (sum(wall) / 1000.0),
        "fps_ci95_low": max(0.0, mean_repeat_fps - critical * sem),
        "fps_ci95_high": mean_repeat_fps + critical * sem,
        "frame_time_ms": statistics.mean(wall),
        "p95_frame_time_ms": percentile(0.95),
        "p99_frame_time_ms": percentile(0.99),
        "peak_vram_mb": speed.get("nvml_peak_vram_mb"),
        "scene_load_time_ms": speed.get("scene_load_time_ms"),
        "time_to_first_frame_ms": speed.get("time_to_first_frame_ms"),
    }


def collect(
    run_dir: Path,
    decoded_ply: Path,
    case_id: str,
    codec_id: str,
    artifact_manifest_path: Path | None,
    reference_result_path: Path | None,
    visual_audit: str,
) -> tuple[dict, dict]:
    suite = _load(ROOT / "benchmark" / "suite.json")
    protocol = _load(ROOT / "benchmark" / "protocol.json")
    case = next(row for row in suite["cases"] if row["case_id"] == case_id)
    canonical_ply = ROOT / case["scene_path"]
    cameras = ROOT / case["camera_path"]
    ground_truth = ROOT / case["ground_truth_path"]
    canonical_sha = _sha256(canonical_ply)
    decoded_sha = _sha256(decoded_ply)
    if canonical_sha != case["canonical_assets"]["checkpoint_sha256"]:
        raise ValueError("canonical checkpoint hash mismatch")

    speed_path = run_dir / "gsplat" / "speed" / "benchmark_results.json"
    quality_path = run_dir / "gsplat" / "quality" / "quality_gt.json"
    nvml_path = run_dir / "gsplat" / "speed" / "nvml_samples.json"
    speed_document, quality_document, nvml_document = _load(speed_path), _load(quality_path), _load(nvml_path)
    if speed_document.get("status") != "complete":
        raise ValueError("compression speed run is incomplete")
    speed, quality_row = _speed_row(speed_document, "gsplat"), _quality_row(quality_document, "gsplat")
    if speed_document.get("scene", {}).get("model_sha256") != decoded_sha:
        raise ValueError("speed run decoded checkpoint hash mismatch")
    if quality_document.get("scene_sha256") != decoded_sha:
        raise ValueError("quality run decoded checkpoint hash mismatch")
    expected_camera_sha = _sha256(cameras)
    if speed_document.get("scene", {}).get("camera_sha256") != expected_camera_sha:
        raise ValueError("speed run camera hash mismatch")
    if quality_document.get("reference", {}).get("camera_manifest_sha256") != expected_camera_sha:
        raise ValueError("quality run camera hash mismatch")
    quality_frames = quality_row.get("frames", [])
    if [row.get("image") for row in quality_frames] != _expected_quality_images(cameras, ground_truth):
        raise ValueError("compression quality views do not match the canonical camera order")
    render_outputs = _verify_render_outputs(quality_document, quality_frames)
    nvml_samples = nvml_document.get("renderers", {}).get("gsplat", [])
    if not nvml_samples or speed.get("nvml_peak_vram_mb") is None:
        raise ValueError("compression result requires raw NVML process-memory samples")

    quality_values = {
        "psnr_db": quality_row["quality"]["mean_psnr_db"],
        "ssim": quality_row["quality"]["mean_ssim"],
        "lpips": quality_row["quality"]["mean_lpips"],
    }
    if codec_id == "reference-ply":
        artifact = {
            "source_sha256": canonical_sha, "source_bytes": canonical_ply.stat().st_size,
            "compressed_sha256": canonical_sha, "compressed_bytes": canonical_ply.stat().st_size,
            "decoded_sha256": decoded_sha, "decoded_bytes": decoded_ply.stat().st_size,
            "compression_ratio": 1.0, "encode_ms": 0.0, "decode_ms": 0.0,
            "cpu_only_decode": True, "peak_decode_vram_mb": 0.0,
        }
        reference_quality = quality_values
        reference_result = None
    else:
        if artifact_manifest_path is None or reference_result_path is None:
            raise ValueError("compressed rows require artifact and reference result paths")
        manifest = _load(artifact_manifest_path)
        if manifest["source"]["sha256"] != canonical_sha:
            raise ValueError("compression artifact source hash mismatch")
        if manifest["compressed_artifact"]["sha256"] != manifest["decode"]["artifact_sha256"]:
            raise ValueError("compression artifact decode identity mismatch")
        reference_result = _load(reference_result_path)
        if reference_result["case"]["case_id"] != case_id or reference_result["codec"]["id"] != "reference-ply":
            raise ValueError("invalid compression reference result")
        reference_quality = reference_result["metrics"]["quality"]
        artifact = {
            "source_sha256": manifest["source"]["sha256"],
            "source_bytes": manifest["source"]["bytes"],
            "compressed_sha256": manifest["compressed_artifact"]["sha256"],
            "compressed_bytes": manifest["compressed_artifact"]["bytes"],
            "decoded_sha256": decoded_sha, "decoded_bytes": decoded_ply.stat().st_size,
            "compression_ratio": manifest["compressed_artifact"]["compression_ratio"],
            "encode_ms": manifest["timings_ms"]["encode"],
            "decode_ms": manifest["timings_ms"]["decode_validation"],
            "cpu_only_decode": manifest["decode"]["cpu_only"],
            "peak_decode_vram_mb": manifest["decode"]["peak_decode_vram_mb"],
        }

    environment = _environment(speed_document, speed)
    if reference_result is not None:
        reference_environment = reference_result["environment"]
        for key in ("gpu_uuid", "driver", "cuda_runtime", "pytorch"):
            if environment.get(key) != reference_environment.get(key):
                raise ValueError(f"compression reference cohort mismatch: {key}")
    delta, gate = quality_gate(reference_quality, quality_values, visual_audit)
    raw = {
        "schema_version": 1,
        "wall_frame_times_ms": speed["wall_frame_times_ms"],
        "repeat_wall_frame_times_ms": speed["repeat_wall_frame_times_ms"],
        "gpu_frame_times_ms": speed.get("frame_times_ms", []),
        "repeat_gpu_frame_times_ms": speed.get("repeat_frame_times_ms", []),
        "nvml_process_memory_samples": nvml_samples,
        "quality_frames": quality_frames,
        "render_output_root": quality_document["render_outputs"]["root"],
        "render_outputs": render_outputs,
        "source_speed_sha256": _sha256(speed_path),
        "source_quality_sha256": _sha256(quality_path),
        "source_nvml_sha256": _sha256(nvml_path),
    }
    measured_at = datetime.now(timezone.utc).isoformat()
    identity = f"{case_id}|{codec_id}|{decoded_sha}|{measured_at}|{_git_commit()}"
    result = {
        "schema_version": "1.0", "result_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "evidence_tier": "measured", "status": "complete",
        "case": {
            "case_id": case_id, "dataset_id": case["dataset_id"], "scene_id": case["scene_id"],
            "canonical_checkpoint_sha256": canonical_sha,
            "camera_trajectory_sha256": expected_camera_sha,
            "quality_reference_sha256": case["canonical_assets"]["ground_truth_manifest_sha256"],
            "resolution": protocol["resolution"],
        },
        "codec": {"id": codec_id, "artifact": artifact},
        "renderer": {
            "id": "gsplat", "config_id": "gsplat", "source_commit": speed.get("renderer_commit_hash"),
        },
        "environment": environment,
        "metrics": {
            "performance": _performance(speed), "quality": quality_values,
            "quality_delta": delta, "near_lossless_gate": gate,
        },
        "raw_samples": raw,
        "provenance": {
            "benchmark_commit": _git_commit(), "measured_at": measured_at,
            "run_dir": str(run_dir.resolve()),
            "artifact_manifest": str(artifact_manifest_path.resolve()) if artifact_manifest_path else None,
            "reference_result": str(reference_result_path.resolve()) if reference_result_path else None,
        },
    }
    schema = _load(ROOT / "benchmark" / "schemas" / "compression-result.schema.json")
    validate_schema(result, schema)
    return result, raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--decoded-ply", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--codec-id", choices=["reference-ply", "block-float", "tile-codebook"], required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--visual-audit", choices=["not_applicable", "pending", "pass", "fail"], default="pending")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result, raw = collect(
        args.run_dir.resolve(), args.decoded_ply.resolve(), args.case_id, args.codec_id,
        args.artifact_manifest.resolve() if args.artifact_manifest else None,
        args.reference_result.resolve() if args.reference_result else None,
        args.visual_audit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_samples.json"
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["raw_samples"] = {"uri": str(raw_path), "sha256": _sha256(raw_path)}
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
