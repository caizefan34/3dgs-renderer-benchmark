#!/usr/bin/env python
"""Download, checksum, extract, and inventory benchmark datasets."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import tarfile
import urllib.request
import uuid
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _CRC32C:
    def __init__(self) -> None:
        self._value = 0xFFFFFFFF

    def update(self, data: bytes) -> None:
        value = self._value
        for byte in data:
            value ^= byte
            for _ in range(8):
                value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        self._value = value

    def value(self) -> int:
        return self._value ^ 0xFFFFFFFF


def _crc32c(data: bytes) -> int:
    digest = _CRC32C()
    digest.update(data)
    return digest.value()


def _file_verification(path: Path, expected: dict) -> dict:
    sha = hashlib.sha256()
    md5 = hashlib.md5() if expected.get("md5_base64") else None
    crc = None
    if expected.get("crc32c_base64"):
        try:
            import google_crc32c
        except ImportError as exc:
            raise RuntimeError(
                "google-crc32c is required to verify GCS CRC32C metadata"
            ) from exc
        crc = google_crc32c.Checksum()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            sha.update(chunk)
            if md5:
                md5.update(chunk)
            if crc:
                crc.update(chunk)
    return {
        "size_bytes": size,
        "sha256": sha.hexdigest(),
        "md5_base64": base64.b64encode(md5.digest()).decode("ascii") if md5 else None,
        "crc32c_base64": base64.b64encode(crc.digest()).decode("ascii") if crc else None,
    }


def _verify_source(path: Path, source: dict, label: str) -> dict:
    expected = source.get("checksums", {})
    if not source.get("size_bytes"):
        raise ValueError(f"{label}: source size is not pinned")
    if not expected.get("sha256") and not (
        expected.get("md5_base64") and expected.get("crc32c_base64")
    ):
        raise ValueError(f"{label}: source checksums are not pinned")
    actual = _file_verification(path, expected)
    if actual["size_bytes"] != source["size_bytes"]:
        raise ValueError(
            f"{label}: size mismatch: expected {source['size_bytes']}, got {actual['size_bytes']}"
        )
    labels = {
        "sha256": "SHA-256",
        "md5_base64": "MD5",
        "crc32c_base64": "CRC32C",
    }
    for key, display in labels.items():
        if expected.get(key) and actual[key] != expected[key]:
            raise ValueError(
                f"{label}: {display} mismatch: expected {expected[key]}, got {actual[key]}"
            )
    return actual


def _safe_extract_zip(
    archive: Path, destination: Path, archive_root: str | None = None
) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes destination: {member.filename}")
        prefix = archive_root.strip("/") + "/" if archive_root else ""
        selected = 0
        for member in source.infolist():
            if prefix and not member.filename.startswith(prefix):
                continue
            relative = member.filename[len(prefix):] if prefix else member.filename
            if not relative or member.is_dir():
                continue
            target = (destination / relative).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes destination: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=8 * 1024 * 1024)
            selected += 1
        if archive_root and not selected:
            raise ValueError(f"archive contains no files under {archive_root!r}")


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes destination: {member.name}")
        source.extractall(destination)


def _legacy_scene(manifest: dict) -> tuple[str, dict]:
    expected = manifest.get("archive_sha256")
    if not expected:
        raise ValueError(
            f"{manifest['dataset_id']}: archive_sha256 is not pinned; Tier A preparation is disabled"
        )
    return manifest["dataset_id"], {
        "source": {
            "url": manifest.get("archive_url"),
            "size_bytes": manifest.get("archive_size_bytes"),
            "checksums": {"sha256": expected},
            "archive_format": manifest.get("archive_format"),
        },
        "archive_root": None,
    }


def prepare(
    manifest_path: Path,
    data_root: Path,
    archive_override: Path | None = None,
    scene_id: str | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = manifest["dataset_id"]
    if isinstance(manifest.get("scenes"), dict):
        if scene_id is None:
            if len(manifest["scenes"]) != 1:
                raise ValueError(f"{dataset_id}: select one scene")
            scene_id = next(iter(manifest["scenes"]))
        try:
            scene = manifest["scenes"][scene_id]
        except KeyError as exc:
            raise ValueError(f"{dataset_id}: unknown scene {scene_id!r}") from exc
        raw = data_root / "raw" / dataset_id / scene_id
    else:
        scene_id, scene = _legacy_scene(manifest)
        raw = data_root / "raw" / dataset_id
    source = scene["source"]
    downloads = data_root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_format = source.get("archive_format")
    suffix = ".zip" if archive_format == "zip" else ".tar"
    archive = archive_override or downloads / source.get(
        "filename", f"{dataset_id}-{scene_id}{suffix}"
    )
    if not archive.exists():
        url = source.get("url")
        if not url:
            raise FileNotFoundError(
                f"{dataset_id}/{scene_id}: place the official archive at {archive} or pin an official URL"
            )
        partial = archive.with_suffix(archive.suffix + ".partial")
        urllib.request.urlretrieve(url, partial)
        try:
            _verify_source(partial, source, f"{dataset_id}/{scene_id}")
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(archive)

    verification = _verify_source(archive, source, f"{dataset_id}/{scene_id}")
    inventory_path = raw / "dataset_inventory.json"
    if raw.exists():
        if inventory_path.exists():
            existing = json.loads(inventory_path.read_text(encoding="utf-8"))
            if existing.get("source_archive_sha256") == verification["sha256"]:
                return existing
        if any(raw.iterdir()):
            raise ValueError(f"{raw} is non-empty but was not prepared from the pinned source")

    raw.parent.mkdir(parents=True, exist_ok=True)
    staging = raw.parent / f".{raw.name}.stage-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        if archive_format == "zip":
            _safe_extract_zip(archive, staging, scene.get("archive_root"))
        elif archive_format in {"tar", "tar.gz", "tgz"}:
            if scene.get("archive_root"):
                raise ValueError("archive_root selection is only implemented for zip sources")
            _safe_extract_tar(archive, staging)
        else:
            raise ValueError(f"unsupported archive format: {archive_format}")
        inventory = {
            "schema_version": 2,
            "dataset_id": dataset_id,
            "scene_id": scene_id,
            "source_url": source.get("url"),
            "source_generation": source.get("generation"),
            "source_archive": str(archive.resolve()),
            "source_archive_sha256": verification["sha256"],
            "source_verification": verification,
            "raw_root": str(raw.resolve()),
            "files": sorted(
                str(path.relative_to(staging)).replace("\\", "/")
                for path in staging.rglob("*") if path.is_file()
            ),
            "image_directory": scene.get("image_directory"),
            "conversion_status": "raw_verified_canonical_case_preparation_required",
        }
        (staging / "dataset_inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(raw)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dataset id or all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--scene")
    args = parser.parse_args()
    manifests = sorted((ROOT / "benchmark" / "datasets").glob("*.json"))
    selected = manifests if args.dataset == "all" else [ROOT / "benchmark" / "datasets" / f"{args.dataset}.json"]
    if args.archive and len(selected) != 1:
        raise SystemExit("--archive requires one dataset")
    for path in selected:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        scene_ids = [args.scene] if args.scene else (
            list(manifest["scenes"]) if isinstance(manifest.get("scenes"), dict) else [None]
        )
        if args.archive and len(scene_ids) > 1:
            sources = [manifest["scenes"][scene]["source"] for scene in scene_ids]
            identities = {(source.get("url"), source.get("filename")) for source in sources}
            if len(identities) > 1:
                raise SystemExit("--archive requires --scene when scenes use different sources")
        for scene_id in scene_ids:
            result = prepare(path, args.data_root, args.archive, scene_id)
            print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
