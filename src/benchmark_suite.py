"""Official benchmark-suite loading and asset validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_PATH = REPO_ROOT / "benchmark_suite" / "suite.json"
BENCHMARK_SUITE_VERSION = "v2.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file_hash(path, expected_sha256: str, label: str) -> str:
    """Require an asset to exist and match its pinned SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Official {label} asset not found: {path}")
    actual = _sha256(path)
    if actual != expected_sha256.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_benchmark_suite(path=DEFAULT_SUITE_PATH) -> dict:
    """Load and minimally validate the committed official suite manifest."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        suite = json.load(handle)
    if suite.get("version") != BENCHMARK_SUITE_VERSION:
        raise ValueError(
            f"Unsupported benchmark suite version: {suite.get('version')!r}"
        )
    if not suite.get("suite_id") or not suite.get("scenes"):
        raise ValueError("Benchmark suite requires suite_id and scenes")
    scene_ids = [scene.get("scene_id") for scene in suite["scenes"]]
    if None in scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Benchmark suite scene IDs must be present and unique")
    for name, resolution in suite.get("resolution_profiles", {}).items():
        if (
            not name
            or not isinstance(resolution, list)
            or len(resolution) != 2
            or any(not isinstance(value, int) or value <= 0 for value in resolution)
        ):
            raise ValueError(f"Invalid resolution profile: {name!r}")
    return suite


def _asset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_suite_case(
    scene_id: str,
    resolution_profile: str,
    *,
    suite_path=DEFAULT_SUITE_PATH,
    verify_assets: bool = True,
) -> dict:
    """Resolve one official scene/resolution case and optionally verify assets."""
    suite = load_benchmark_suite(suite_path)
    try:
        scene = next(item for item in suite["scenes"] if item["scene_id"] == scene_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown official benchmark scene: {scene_id}") from exc
    try:
        resolution = suite["resolution_profiles"][resolution_profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown official resolution profile: {resolution_profile}"
        ) from exc

    paths = {
        "dataset": _asset_path(scene["dataset_path"]),
        "scene": _asset_path(scene["scene_path"]),
        "camera": _asset_path(scene["camera_path"]),
    }
    if verify_assets:
        for label, path in paths.items():
            validate_file_hash(path, scene[f"{label}_sha256"], label)

    return {
        "official": True,
        "validated": bool(verify_assets),
        "suite_id": suite["suite_id"],
        "suite_version": suite["version"],
        "suite_case_id": f"{scene_id}@{resolution_profile}",
        "dataset_id": scene["dataset_id"],
        "dataset_family": scene["dataset_family"],
        "dataset_sha256": scene["dataset_sha256"],
        "scene_id": scene_id,
        "scene_sha256": scene["scene_sha256"],
        "camera_sha256": scene["camera_sha256"],
        "resolution_profile": resolution_profile,
        "resolution": list(resolution),
        "protocol": dict(suite["protocol"]),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def official_metadata_matches_suite(metadata: dict) -> bool:
    """Return whether result metadata exactly matches a registered suite case."""
    if not metadata.get("official") or not metadata.get("validated"):
        return False
    case_id = metadata.get("suite_case_id", "")
    if "@" not in case_id:
        return False
    scene_id, resolution_profile = case_id.rsplit("@", 1)
    try:
        expected = resolve_suite_case(
            scene_id, resolution_profile, verify_assets=False
        )
    except (KeyError, TypeError, ValueError):
        return False
    keys = (
        "suite_id", "suite_version", "suite_case_id", "dataset_sha256",
        "scene_sha256", "camera_sha256", "resolution_profile", "resolution",
    )
    return all(metadata.get(key) == expected.get(key) for key in keys)

