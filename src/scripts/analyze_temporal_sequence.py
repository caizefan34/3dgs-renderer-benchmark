#!/usr/bin/env python
"""Measure ordered-camera temporal residual from retained render evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from schema_validation import validate_schema  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    root = next(
        (parent for parent in path.parents if (parent / "benchmark" / "suite.json").is_file()),
        ROOT,
    )
    return root / candidate


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def analyze(metrics_path: Path, ground_truth_dir: Path) -> dict:
    source = _load(metrics_path)
    raw_record = source["metrics"]["raw_samples"]
    raw_path = _repo_relative(metrics_path, raw_record["uri"])
    if _sha256(raw_path) != raw_record["sha256"]:
        raise ValueError("source result raw-sample hash mismatch")
    raw = _load(raw_path)
    render_root = Path(raw["render_output_root"])
    outputs = {row["image"]: row for row in raw["render_outputs"]}
    ordered_images = [row["image"] for row in raw["quality_frames"]]
    if len(ordered_images) < 2 or len(set(ordered_images)) != len(ordered_images):
        raise ValueError("temporal analysis requires at least two unique ordered frames")
    ground_truth_index = {
        path.stem: path for path in ground_truth_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    residual_means = []
    residual_pixels = []
    luma_means = []
    squared_error_sum = 0.0
    value_count = 0
    previous_prediction = previous_reference = None
    for image_name in ordered_images:
        prediction = _read_rgb(render_root / outputs[image_name]["path"])
        reference_path = ground_truth_index.get(Path(image_name).stem)
        if reference_path is None:
            raise FileNotFoundError(f"missing ordered GT frame: {image_name}")
        reference = _read_rgb(reference_path)
        if prediction.shape != reference.shape:
            raise ValueError(f"prediction/GT shape mismatch: {image_name}")
        if previous_prediction is not None:
            residual = (prediction - previous_prediction) - (reference - previous_reference)
            absolute = np.abs(residual)
            per_pixel = absolute.mean(axis=2)
            residual_means.append(float(per_pixel.mean()))
            residual_pixels.append(per_pixel.reshape(-1))
            luma = np.abs(
                residual[..., 0] * 0.2126
                + residual[..., 1] * 0.7152
                + residual[..., 2] * 0.0722
            )
            luma_means.append(float(luma.mean()))
            squared_error_sum += float(np.sum(residual * residual))
            value_count += residual.size
        previous_prediction, previous_reference = prediction, reference
    all_pixels = np.concatenate(residual_pixels)
    mse = squared_error_sum / value_count
    temporal_psnr = 999.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)
    return {
        "schema_version": "1.0", "status": "complete",
        "evidence_tier": source["evidence_tier"],
        "source_result": {
            "result_id": source["result_id"],
            "renderer_id": source["renderer"]["id"],
            "config_id": source["renderer"]["config_id"],
            "case_id": source["benchmark"]["case_id"],
            "metrics_sha256": _sha256(metrics_path),
            "raw_samples_sha256": raw_record["sha256"],
        },
        "sequence": {
            "frame_count": len(ordered_images),
            "transition_count": len(ordered_images) - 1,
            "camera_order": "canonical evaluation manifest order",
            "metric_scope": "adjacent-frame RGB delta residual; no optical-flow compensation",
        },
        "metrics": {
            "mean_temporal_residual": float(np.mean(residual_means)),
            "p95_temporal_residual": float(np.percentile(all_pixels, 95)),
            "mean_luma_temporal_residual": float(np.mean(luma_means)),
            "temporal_delta_psnr_db": temporal_psnr,
        },
        "provenance": {
            "source_metrics": str(metrics_path),
            "ground_truth_dir": str(ground_truth_dir),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.metrics.resolve(), args.ground_truth_dir.resolve())
    validate_schema(
        result,
        _load(ROOT / "benchmark" / "schemas" / "temporal-result.schema.json"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
