"""Deterministic, self-verifying evidence bundle for archival releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


class ReleaseBundleError(ValueError):
    """Raised when a release bundle cannot be built reproducibly."""


def _require_tracked(path: Path, repository_root: Path) -> None:
    relative = path.relative_to(repository_root).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ReleaseBundleError(f"release member is not Git-tracked: {relative}")


def _git_commit(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ReleaseBundleError("cannot resolve HEAD commit")
    return result.stdout.strip()


def _bundle_members(repository_root: Path) -> list[Path]:
    claims = json.loads((repository_root / "paper" / "claims.json").read_text(encoding="utf-8"))
    members = {
        Path("README.md"),
        Path("LICENSE"),
        Path("CITATION.cff"),
        Path("benchmark/suite.json"),
        Path("benchmark/protocol.json"),
        Path("docs/methodology.md"),
        Path("docs/protocol.md"),
        Path("docs/leaderboard/ranking.json"),
        Path("docs/leaderboard/ranking.md"),
        Path("reports/generated/compression-expanded-final/compression-results.json"),
        Path("paper/README.md"),
        Path("paper/claims.json"),
    }
    for claim in claims.get("claims", []):
        if claim.get("status") == "supported":
            for evidence in claim.get("evidence", []):
                members.add(Path(evidence["path"]))
    return sorted(members, key=lambda path: path.as_posix())


def build_release_bundle(repository_root: str | Path, output: str | Path) -> dict:
    """Write a deterministic zip whose manifest hashes every member."""
    root = Path(repository_root).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    members = _bundle_members(root)
    entries = []
    for member in members:
        path = (root / member).resolve()
        if root not in path.parents and path != root:
            raise ReleaseBundleError(f"release member escapes repository root: {member.as_posix()}")
        if not path.is_file():
            raise ReleaseBundleError(f"release member missing: {member.as_posix()}")
        _require_tracked(path, root)
        payload = path.read_bytes()
        entries.append({
            "path": member.as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    manifest = {
        "schema_version": "1.0",
        "deterministic": True,
        "repository_commit": _git_commit(root),
        "files": entries,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo(
            "artifact-manifest.json",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.external_attr = 0
        info.create_system = 3
        archive.writestr(info, manifest_bytes)
        for entry in entries:
            info = zipfile.ZipInfo(entry["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0
            info.create_system = 3
            archive.writestr(info, (root / entry["path"]).read_bytes())

    bundle_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "file_count": len(entries) + 1,
        "bundle_sha256": bundle_hash,
        "bundle_bytes": output_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path for the deterministic zip bundle")
    args = parser.parse_args()
    try:
        summary = build_release_bundle(Path(__file__).resolve().parents[1], args.output)
    except ReleaseBundleError as exc:
        print(f"release bundle failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())