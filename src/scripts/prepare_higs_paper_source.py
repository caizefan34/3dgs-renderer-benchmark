#!/usr/bin/env python
"""Create an auditable gsplat checkout for the HiGS paper experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import audit_gsplat_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("official", "higs"), required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmark" / "higs-paper-protocol.json",
    )
    args = parser.parse_args()
    source_dir = (
        args.source_dir
        or ROOT
        / "artifacts"
        / "renderer-sources"
        / ("gsplat-official" if args.variant == "official" else "gsplat-higs")
    ).resolve()
    if source_dir.exists():
        print(
            f"refusing to overwrite existing source directory: {source_dir}",
            file=sys.stderr,
        )
        return 1

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    spec = protocol["methods"]["gsplat" if args.variant == "official" else "higs_full"]
    patch = ROOT / protocol["methods"]["higs_full"]["patches"][0]
    try:
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", spec["repository"], str(source_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source_dir), "checkout", "--detach", spec["commit"]],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source_dir),
                "submodule",
                "update",
                "--init",
                "--recursive",
            ],
            check=True,
        )
        if args.variant == "higs":
            subprocess.run(
                ["git", "-C", str(source_dir), "apply", str(patch)], check=True
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(
            f"source preparation failed; partial checkout was preserved at "
            f"{source_dir}: {exc}",
            file=sys.stderr,
        )
        return 1

    audit = audit_gsplat_source(source_dir)
    if audit["head_commit"] != spec["commit"] or (
        args.variant == "higs" and not audit["has_higs_dynamic_api"]
    ) or (
        args.variant == "higs"
        and audit["source_state_sha256"] != spec["source_state_sha256"]
    ):
        print(json.dumps(audit, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
