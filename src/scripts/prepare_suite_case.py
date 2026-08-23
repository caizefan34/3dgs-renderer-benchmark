#!/usr/bin/env python
"""Build deterministic 1080p camera and asset identities for one suite case."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _selected_indices(count: int, required: int) -> list[int]:
    if count < required:
        raise ValueError(f"camera export has {count} views; {required} are required")
    if required == 1:
        return [0]
    return [round(index * (count - 1) / (required - 1)) for index in range(required)]


def _center_crop(width: int, height: int, target_width: int, target_height: int) -> list[int]:
    source_aspect = width / height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        crop_height = height
        crop_width = round(height * target_aspect)
        left = (width - crop_width) // 2
        return [left, 0, left + crop_width, height]
    crop_width = width
    crop_height = round(width / target_aspect)
    top = (height - crop_height) // 2
    return [0, top, width, top + crop_height]


def build_camera_candidate(
    source_cameras: list[dict],
    image_sizes: dict[str, tuple[int, int]],
    required_views: int,
    target_width: int,
    target_height: int,
) -> tuple[bytes, list[str]]:
    """Order by image name, select evenly, and center-crop to the target aspect."""
    by_name = {}
    for camera in source_cameras:
        name = camera.get("img_name") or camera.get("image_name")
        if not name or name in by_name:
            raise ValueError("source camera image names must be present and unique")
        by_name[name] = camera
    ordered_names = sorted(by_name, key=str.casefold)
    selected_names = [ordered_names[index] for index in _selected_indices(len(ordered_names), required_views)]
    output = []
    target_aspect = target_width / target_height
    for output_id, name in enumerate(selected_names):
        source = by_name[name]
        if name not in image_sizes:
            raise ValueError(f"camera {name!r} has no source image")
        source_width = int(source["width"])
        source_height = int(source["height"])
        if source_width / source_height > target_aspect:
            crop_width = source_height * target_aspect
            crop_height = float(source_height)
        else:
            crop_width = float(source_width)
            crop_height = source_width / target_aspect
        row = {
            **source,
            "id": output_id,
            "width": target_width,
            "height": target_height,
            "fx": float(source["fx"]) * target_width / crop_width,
            "fy": float(source["fy"]) * target_height / crop_height,
            "reference_crop": _center_crop(*image_sizes[name], target_width, target_height),
            "conversion": "center crop to target aspect, then resize",
        }
        output.append(row)
    payload = (json.dumps(
        output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n").encode("utf-8")
    return payload, selected_names


def canonical_asset_hashes(
    checkpoint: bytes,
    camera_payload: bytes,
    ground_truth_entries: list[dict],
    expected: dict | None = None,
) -> dict:
    gt_payload = json.dumps(
        ground_truth_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual = {
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "camera_sha256": hashlib.sha256(camera_payload).hexdigest(),
        "ground_truth_manifest_sha256": hashlib.sha256(gt_payload).hexdigest(),
    }
    if expected is not None:
        if expected.get("status") != "pinned":
            raise ValueError("suite canonical assets are unpublished")
        mismatches = [key for key, value in actual.items() if expected.get(key) != value]
        if mismatches:
            raise ValueError(f"canonical asset mismatch: {', '.join(mismatches)}")
    return actual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _member(archive, specification: dict):
    info = archive.getinfo(specification["entry"])
    if info.file_size != specification["size_bytes"]:
        raise ValueError(f"model member size mismatch: {specification['entry']}")
    if f"{info.CRC:08x}" != specification["crc32"]:
        raise ValueError(f"model member CRC32 mismatch: {specification['entry']}")
    return info


def _read_member(archive, specification: dict) -> bytes:
    info = _member(archive, specification)
    payload = archive.read(info)
    if specification.get("sha256") and hashlib.sha256(payload).hexdigest() != specification["sha256"]:
        raise ValueError(f"model member SHA-256 mismatch: {specification['entry']}")
    return payload


def _copy_member(archive, specification: dict, destination: Path) -> str:
    info = _member(archive, specification)
    digest = hashlib.sha256()
    with archive.open(info) as source, destination.open("wb") as target:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest()


def _hash_member(archive, specification: dict) -> str:
    info = _member(archive, specification)
    digest = hashlib.sha256()
    with archive.open(info) as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _model_archive(specification: dict, override: Path | None):
    if override is not None:
        if override.stat().st_size != specification["size_bytes"]:
            raise ValueError("official model archive size mismatch")
        with zipfile.ZipFile(override) as archive:
            yield archive
        return
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError(
            "remotezip is required for selective official-model download; "
            "install requirements-benchmark.txt or pass --model-archive"
        ) from exc
    with RemoteZip(specification["url"]) as archive:
        yield archive


@contextlib.contextmanager
def _source_archive(specification: dict):
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError("remotezip is required for --audit-only") from exc
    with RemoteZip(specification["url"]) as archive:
        yield archive


def _image_index(directory: Path) -> dict[str, Path]:
    paths = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    index = {path.stem: path for path in paths}
    if len(index) != len(paths):
        raise ValueError("source image stems must be unique")
    return index


def _image_sizes(index: dict[str, Path]) -> dict[str, tuple[int, int]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for canonical camera preparation") from exc
    sizes = {}
    for name, path in index.items():
        with Image.open(path) as image:
            sizes[name] = image.size
    return sizes


def _verify_raw_inventory(manifest: dict, scene_id: str, inventory: dict) -> None:
    if inventory.get("dataset_id") != manifest["dataset_id"] or inventory.get("scene_id") != scene_id:
        raise ValueError("raw inventory identifies a different dataset scene")
    expected = manifest["scenes"][scene_id]["source"]
    actual = inventory.get("source_verification", {})
    if actual.get("size_bytes") != expected["size_bytes"]:
        raise ValueError("raw inventory source size mismatch")
    for key, value in expected["checksums"].items():
        if actual.get(key) != value:
            raise ValueError(f"raw inventory source checksum mismatch: {key}")


def prepare_case(
    case_id: str,
    model_archive: Path | None = None,
    candidate_output: Path | None = None,
) -> dict:
    suite = _load(ROOT / "benchmark" / "suite.json")
    protocol = _load(ROOT / "benchmark" / "protocol.json")
    case = next((row for row in suite["cases"] if row["case_id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown case: {case_id}")
    manifest = _load(ROOT / case["dataset_manifest"])
    scene_id = case["scene_id"]
    scene_spec = manifest["scenes"][scene_id]
    raw_root = ROOT / "datasets" / "raw" / case["dataset_id"] / scene_id
    inventory_path = raw_root / "dataset_inventory.json"
    if not inventory_path.exists():
        raise ValueError(f"run benchmark prepare {case['dataset_id']} --scene {scene_id} first")
    inventory = _load(inventory_path)
    _verify_raw_inventory(manifest, scene_id, inventory)
    image_dir = raw_root / scene_spec["image_directory"]
    if not image_dir.is_dir():
        raise ValueError(f"prepared source image directory is missing: {image_dir}")

    target = (ROOT / case["scene_path"]).parent
    if target.exists():
        raise ValueError(f"processed case already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.stage-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        model_spec = manifest["official_model_archive"]
        entries = model_spec["scenes"][scene_id]
        with _model_archive(model_spec, model_archive) as archive:
            source_cameras = _read_member(archive, entries["cameras"])
            checkpoint_sha = _copy_member(
                archive, entries["checkpoint"], staging / "point_cloud.ply"
            )
        camera_document = json.loads(source_cameras)
        images = _image_index(image_dir)
        camera_payload, selected_names = build_camera_candidate(
            camera_document,
            _image_sizes(images),
            protocol["timing"]["measured_frames_per_repeat"],
            protocol["resolution"][0],
            protocol["resolution"][1],
        )
        (staging / "eval_cameras.json").write_bytes(camera_payload)
        gt_target = staging / "eval_images"
        gt_target.mkdir()
        gt_entries = []
        for name in selected_names:
            source = images[name]
            destination = gt_target / source.name
            shutil.copy2(source, destination)
            gt_entries.append({"image": destination.name, "sha256": sha256_file(destination)})
        gt_payload = json.dumps(
            gt_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        actual = {
            "checkpoint_sha256": checkpoint_sha,
            "camera_sha256": hashlib.sha256(camera_payload).hexdigest(),
            "ground_truth_manifest_sha256": hashlib.sha256(gt_payload).hexdigest(),
        }
        candidate = {
            "schema_version": 1,
            "case_id": case_id,
            "dataset_id": case["dataset_id"],
            "scene_id": scene_id,
            "canonical_assets": actual,
            "dataset_source_archive_sha256": inventory["source_archive_sha256"],
            "official_model_archive": model_spec["url"],
            "checkpoint_entry": entries["checkpoint"],
            "source_camera_entry": entries["cameras"],
            "source_camera_sha256": hashlib.sha256(source_cameras).hexdigest(),
            "camera_selection": "casefold image-name order; evenly spaced including endpoints",
            "reference_conversion": "center crop to 16:9, then area resize at metric time",
            "view_count": len(selected_names),
        }
        candidate_output = candidate_output or ROOT / "datasets" / "candidates" / f"{case_id}.json"
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        candidate_output.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")

        canonical = case["canonical_assets"]
        if canonical.get("status") != "pinned" or any(
            canonical.get(key) != value for key, value in actual.items()
        ):
            return {"status": "candidate", "path": str(candidate_output), **candidate}

        preparation = {
            **candidate,
            **actual,
            "status": "canonical",
            "source_archive_sha256": inventory["source_archive_sha256"],
            "camera_trajectory_sha256": actual["camera_sha256"],
            "ground_truth_file_manifest_sha256": actual["ground_truth_manifest_sha256"],
            "ground_truth_file_count": len(gt_entries),
            "ground_truth_files": gt_entries,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
        (staging / "preparation.json").write_text(
            json.dumps(preparation, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(target)
        return {"status": "prepared", "path": str(target), **actual}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def audit_case(case_id: str, candidate_output: Path | None = None) -> dict:
    """Hash canonical candidates by selective official ZIP reads without staging Tier A."""
    suite = _load(ROOT / "benchmark" / "suite.json")
    protocol = _load(ROOT / "benchmark" / "protocol.json")
    case = next((row for row in suite["cases"] if row["case_id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown case: {case_id}")
    manifest = _load(ROOT / case["dataset_manifest"])
    scene_id = case["scene_id"]
    model_spec = manifest["official_model_archive"]
    model_entries = model_spec["scenes"][scene_id]
    with _model_archive(model_spec, None) as archive:
        source_cameras = _read_member(archive, model_entries["cameras"])
        checkpoint_sha = _hash_member(archive, model_entries["checkpoint"])
    camera_document = json.loads(source_cameras)
    by_name = {
        camera.get("img_name") or camera.get("image_name"): camera
        for camera in camera_document
    }
    ordered_names = sorted(by_name, key=str.casefold)
    selected_names = [
        ordered_names[index] for index in _selected_indices(
            len(ordered_names), protocol["timing"]["measured_frames_per_repeat"]
        )
    ]

    scene_spec = manifest["scenes"][scene_id]
    prefix = "/".join((
        scene_spec["archive_root"].strip("/"),
        scene_spec["image_directory"].strip("/"),
    )) + "/"
    image_sizes = {}
    gt_entries = []
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --audit-only") from exc
    with _source_archive(scene_spec["source"]) as archive:
        image_entries = {
            Path(info.filename).stem: info
            for info in archive.infolist()
            if info.filename.startswith(prefix) and not info.is_dir()
            and Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        for name in selected_names:
            if name not in image_entries:
                raise ValueError(f"official source archive has no image for camera {name!r}")
            info = image_entries[name]
            payload = archive.read(info)
            with Image.open(io.BytesIO(payload)) as image:
                image_sizes[name] = image.size
            gt_entries.append({
                "image": Path(info.filename).name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    camera_payload, confirmed_names = build_camera_candidate(
        camera_document,
        image_sizes,
        protocol["timing"]["measured_frames_per_repeat"],
        protocol["resolution"][0],
        protocol["resolution"][1],
    )
    if confirmed_names != selected_names:
        raise AssertionError("camera selection changed during candidate construction")
    gt_payload = json.dumps(
        gt_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual = {
        "checkpoint_sha256": checkpoint_sha,
        "camera_sha256": hashlib.sha256(camera_payload).hexdigest(),
        "ground_truth_manifest_sha256": hashlib.sha256(gt_payload).hexdigest(),
    }
    candidate = {
        "schema_version": 1,
        "status": "audit_only_not_tier_a_prepared",
        "case_id": case_id,
        "dataset_id": case["dataset_id"],
        "scene_id": scene_id,
        "canonical_assets": actual,
        "dataset_source": scene_spec["source"],
        "official_model_archive": model_spec["url"],
        "checkpoint_entry": model_entries["checkpoint"],
        "source_camera_entry": model_entries["cameras"],
        "source_camera_sha256": hashlib.sha256(source_cameras).hexdigest(),
        "camera_selection": "casefold image-name order; evenly spaced including endpoints",
        "reference_conversion": "center crop to 16:9, then area resize at metric time",
        "view_count": len(selected_names),
    }
    candidate_output = candidate_output or ROOT / "datasets" / "candidates" / f"{case_id}.json"
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    return {"path": str(candidate_output), **candidate}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--model-archive", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        if args.model_archive:
            parser.error("--audit-only uses the pinned remote archives")
        result = audit_case(args.case_id, args.candidate_output)
    else:
        result = prepare_case(args.case_id, args.model_archive, args.candidate_output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
