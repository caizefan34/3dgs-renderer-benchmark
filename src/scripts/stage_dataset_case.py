#!/usr/bin/env python
"""Normalize verified checkpoint, cameras, and GT images into one suite case."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--derivation-manifest", type=Path, required=True)
    args = parser.parse_args()

    suite = json.loads((ROOT / "benchmark" / "suite.json").read_text(encoding="utf-8"))
    case = next((row for row in suite["cases"] if row["case_id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"unknown case: {args.case_id}")
    manifest = json.loads((ROOT / case["dataset_manifest"]).read_text(encoding="utf-8"))
    scene_spec = manifest.get("scenes", {}).get(case["scene_id"])
    if not scene_spec:
        raise SystemExit("dataset scene source is not pinned")
    inventory_path = (
        ROOT / "datasets" / "raw" / case["dataset_id"] /
        case["scene_id"] / "dataset_inventory.json"
    )
    if not inventory_path.exists():
        raise SystemExit(f"run benchmark prepare {case['dataset_id']} first")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    verification = inventory.get("source_verification", {})
    if verification.get("size_bytes") != scene_spec["source"].get("size_bytes"):
        raise SystemExit("raw dataset inventory does not match the pinned source size")
    for key, expected in scene_spec["source"].get("checksums", {}).items():
        if verification.get(key) != expected:
            raise SystemExit(f"raw dataset inventory does not match the pinned {key}")
    archive_sha = inventory.get("source_archive_sha256")
    if not archive_sha:
        raise SystemExit("raw dataset inventory has no local SHA-256")

    for source in (args.checkpoint, args.cameras, args.ground_truth_dir, args.derivation_manifest):
        if not source.exists():
            raise SystemExit(f"missing source: {source}")
    derivation = json.loads(args.derivation_manifest.read_text(encoding="utf-8"))
    required_derivation = {
        "training_repository", "training_commit", "training_command", "export_command",
        "dataset_archive_sha256", "checkpoint_sha256", "camera_manifest_sha256",
    }
    missing_derivation = sorted(required_derivation - derivation.keys())
    if missing_derivation:
        raise SystemExit(f"derivation manifest missing: {', '.join(missing_derivation)}")
    if derivation["dataset_archive_sha256"] != archive_sha:
        raise SystemExit("derivation manifest dataset hash mismatch")
    if derivation["checkpoint_sha256"] != sha256(args.checkpoint):
        raise SystemExit("derivation manifest checkpoint hash mismatch")
    if derivation["camera_manifest_sha256"] != sha256(args.cameras):
        raise SystemExit("derivation manifest camera hash mismatch")
    images = sorted(
        path for path in args.ground_truth_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise SystemExit("ground-truth directory has no supported images")
    image_index = {path.stem: path for path in images}
    if len(image_index) != len(images):
        raise SystemExit("ground-truth image stems must be unique")
    camera_document = json.loads(args.cameras.read_text(encoding="utf-8"))
    camera_rows = camera_document if isinstance(camera_document, list) else camera_document.get("cameras", [])
    required_views = json.loads(
        (ROOT / "benchmark" / "protocol.json").read_text(encoding="utf-8")
    )["timing"]["measured_frames_per_repeat"]
    if len(camera_rows) < required_views:
        raise SystemExit(f"camera manifest has {len(camera_rows)} views; {required_views} are required")
    if required_views == 1:
        selected_indices = [0]
    else:
        selected_indices = [round(index * (len(camera_rows) - 1) / (required_views - 1)) for index in range(required_views)]
    selected_cameras = [camera_rows[index] for index in selected_indices]
    selected_images = []
    for camera in selected_cameras:
        image_name = camera.get("image_name") or camera.get("img_name")
        if not image_name or Path(image_name).stem not in image_index:
            raise SystemExit(f"no GT image for selected camera {image_name!r}")
        selected_images.append(image_index[Path(image_name).stem])

    target = (ROOT / case["scene_path"]).parent
    if target.exists():
        raise SystemExit(f"processed case already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.stage-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        shutil.copy2(args.checkpoint, staging / "point_cloud.ply")
        selected_document = selected_cameras if isinstance(camera_document, list) else {**camera_document, "cameras": selected_cameras}
        (staging / "eval_cameras.json").write_text(
            json.dumps(selected_document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        gt_target = staging / "eval_images"
        gt_target.mkdir()
        for image in selected_images:
            shutil.copy2(image, gt_target / image.name)
        gt_entries = [{"image": image.name, "sha256": sha256(gt_target / image.name)} for image in selected_images]
        gt_hash = hashlib.sha256(json.dumps(
            gt_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        preparation = {
            "schema_version": 1,
            "case_id": args.case_id,
            "dataset_id": case["dataset_id"],
            "archive_sha256": archive_sha,
            "source_archive_sha256": archive_sha,
            "checkpoint_sha256": sha256(staging / "point_cloud.ply"),
            "camera_trajectory_sha256": sha256(staging / "eval_cameras.json"),
            "ground_truth_file_manifest_sha256": gt_hash,
            "ground_truth_file_count": len(selected_images),
            "camera_selection": {
                "source_view_count": len(camera_rows),
                "selected_view_count": len(selected_cameras),
                "rule": "evenly spaced ordered indices including first and last",
                "indices": selected_indices
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "conversion": "canonical PLY/cameras.json/eval-image layout; no lossy transformation",
            "derivation": derivation,
        }
        canonical = case["canonical_assets"]
        actual_assets = {
            "checkpoint_sha256": preparation["checkpoint_sha256"],
            "camera_sha256": preparation["camera_trajectory_sha256"],
            "ground_truth_manifest_sha256": preparation["ground_truth_file_manifest_sha256"],
        }
        if canonical.get("status") != "pinned" or any(
            canonical.get(key) != value for key, value in actual_assets.items()
        ):
            raise ValueError(
                "suite canonical assets are unpublished or do not match; "
                f"review and pin this candidate: {json.dumps(actual_assets, sort_keys=True)}"
            )
        (staging / "preparation.json").write_text(
            json.dumps(preparation, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(target)
    except Exception:
        if staging.exists() and staging.parent == target.parent:
            shutil.rmtree(staging)
        raise
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
