"""Installed console entry point for a source-checkout benchmark."""
from __future__ import annotations

import os
from pathlib import Path

import benchmark_cli


def _is_repository_root(path: Path) -> bool:
    return all(
        candidate.is_file()
        for candidate in (
            path / "benchmark" / "protocol.json",
            path / "benchmark" / "suite.json",
            path / "benchmark" / "renderers.json",
            path / "src" / "benchmark_cli.py",
        )
    )


def find_repository_root(start: Path | None = None) -> Path:
    configured = os.environ.get("GSBENCH_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _is_repository_root(root):
            raise SystemExit(
                f"GSBENCH_ROOT is not a 3dgs-renderer-benchmark checkout: {root}"
            )
        return root

    origin = (start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if _is_repository_root(candidate):
            return candidate
    raise SystemExit(
        "benchmark commands require a source checkout; run from the repository "
        "or set GSBENCH_ROOT to its absolute path"
    )


def main(argv=None) -> int:
    benchmark_cli.ROOT = find_repository_root()
    return benchmark_cli.main(argv)
