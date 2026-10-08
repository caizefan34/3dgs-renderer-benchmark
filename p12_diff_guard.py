"""Classify P12 source changes and reject unapproved renderer differences."""

import argparse
import json
from pathlib import Path
import sys

from p12_run_guard import sha256_file


SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".py"}
CUDA_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
METADATA_NAMES = {"pyproject.toml", "setup.py", "setup.cfg", "CMakeLists.txt", "Makefile", "build.ninja", "requirements.txt"}


def _files(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"source tree missing: {root}")
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and ".git" not in path.relative_to(root).parts and "__pycache__" not in path.parts}


def _classify(path, left_hash, right_hash, label, f9_allowlist):
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    if label == "A0_vs_A1" and f9_allowlist.get(path) == {"a0_sha256": left_hash, "a1_sha256": right_hash}:
        return "F9 intended diff"
    if name in METADATA_NAMES or path.startswith("build/"):
        return "build metadata"
    if suffix == ".py" and ("loss" in path.lower() or "trainer" in path.lower()):
        return "loss plumbing"
    return "unexpected"


def _renderer_source(path):
    parts = [part.lower() for part in Path(path).parts]
    name = Path(path).name.lower()
    return Path(path).suffix.lower() in CUDA_SUFFIXES or (
        Path(path).suffix.lower() == ".py" and
        (any(part in {"gsplat", "cuda", "render", "renderer", "rasterizer", "projection", "experimental"} for part in parts)
         or any(token in name for token in ("raster", "render", "projection")))
    )


def compare_trees(left, right, label, f9_allowlist=None):
    """Compare complete trees; explicit content hashes gate every intended F9 change."""
    f9_allowlist = f9_allowlist or {}
    left_files, right_files = _files(left), _files(right)
    changes, errors = [], []
    for path in sorted(left_files.keys() | right_files.keys()):
        old_hash = sha256_file(left_files[path]) if path in left_files else None
        new_hash = sha256_file(right_files[path]) if path in right_files else None
        if old_hash == new_hash:
            continue
        category = _classify(path, old_hash, new_hash, label, f9_allowlist)
        changes.append({"path": path, "category": category, "old_sha256": old_hash, "new_sha256": new_hash})
        suffix = Path(path).suffix.lower()
        if category == "unexpected":
            errors.append(f"{label}: unexpected file change {path}")
        if label == "A0_vs_A1" and category in ("loss plumbing", "build metadata"):
            errors.append(f"{label}: difference outside frozen F9 change: {path}")
        if label != "A0_vs_A1" and _renderer_source(path):
            errors.append(f"{label}: frozen historical renderer source differs: {path}")
        if label == "A0_vs_A1" and suffix in SOURCE_SUFFIXES and category != "F9 intended diff":
            errors.append(f"{label}: algorithmic difference outside approved F9 hashes: {path}")
    return {"comparison": label, "changes": changes, "violations": list(dict.fromkeys(errors))}


def audit(p8_gmain, p12_gmain, p8_c0, p12_c0, a0, a1, f9_allowlist):
    comparisons = [
        compare_trees(p8_gmain, p12_gmain, "P8_GMAIN_vs_P12_GMAIN"),
        compare_trees(p8_c0, p12_c0, "P8_C0_vs_P12_C0"),
        compare_trees(a0, a1, "A0_vs_A1", f9_allowlist),
    ]
    errors = [error for comparison in comparisons for error in comparison["violations"]]
    return {"status": "PASS" if not errors else "FAIL", "comparisons": comparisons, "violations": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p8-gmain", "p12-gmain", "p8-c0", "p12-c0", "a0", "a1"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--f9-allowlist", type=Path, required=True, help="JSON path -> {a0_sha256,a1_sha256}; pin approved F9 changes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        allowlist = json.loads(args.f9_allowlist.read_text(encoding="utf-8"))
        if not isinstance(allowlist, dict):
            raise ValueError("F9 allowlist must be an object")
        result = audit(args.p8_gmain, args.p12_gmain, args.p8_c0, args.p12_c0, args.a0, args.a1, allowlist)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"P12 diff guard input: {exc}", file=sys.stderr)
        return 2
    for error in result["violations"]:
        print(f"INVARIANT VIOLATION: {error}", file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
