#!/usr/bin/env python
"""Create an auditable Speedy-Splat checkout for the HiGS paper experiments."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import audit_speedy_splat_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmark" / "higs-paper-protocol.json",
    )
    args = parser.parse_args()
    source_dir = (
        args.source_dir
        or ROOT / "artifacts" / "renderer-sources" / "speedy-splat"
    ).resolve()
    if source_dir.exists():
        print(
            f"refusing to overwrite existing source directory: {source_dir}",
            file=sys.stderr,
        )
        return 1

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    spec = protocol["methods"]["speedy_splat"]
    patches = [ROOT / p for p in spec["patches"]]
    simple_knn_patch = (
        ROOT / spec["simple_knn_patch"] if spec.get("simple_knn_patch") else None
    )
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
                "submodules/diff-gaussian-rasterization",
                "submodules/simple-knn",
            ],
            check=True,
        )
        for patch in patches:
            subprocess.run(
                ["git", "-C", str(source_dir), "apply", str(patch)], check=True
            )
        if simple_knn_patch is not None:
            submodule = source_dir / "submodules" / "simple-knn"
            subprocess.run(
                ["git", "-C", str(submodule), "apply", str(simple_knn_patch)],
                check=True,
            )
            env = dict(os.environ)
            env.update(
                {
                    "GIT_AUTHOR_NAME": "Codex Bench",
                    "GIT_AUTHOR_EMAIL": "codex-bench@local",
                    "GIT_AUTHOR_DATE": "2026-08-07 09:27:19 +0800",
                    "GIT_COMMITTER_NAME": "Codex Bench",
                    "GIT_COMMITTER_EMAIL": "codex-bench@local",
                    "GIT_COMMITTER_DATE": "2026-08-07 09:27:19 +0800",
                }
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(submodule),
                    "commit",
                    "-q",
                    "-m",
                    "Add <cfloat> for FLT_MAX (CUDA >= 12.6 toolchains)",
                ],
                check=True,
                env=env,
            )
            subprocess.run(
                ["git", "-C", str(source_dir), "add", "submodules/simple-knn"],
                check=True,
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(
            f"source preparation failed; partial checkout was preserved at "
            f"{source_dir}: {exc}",
            file=sys.stderr,
        )
        return 1

    audit = audit_speedy_splat_source(source_dir)
    if (
        audit["head_commit"] != spec["commit"]
        or audit["source_state_sha256"] != spec["source_state_sha256"]
        or audit["trainer_sha256"] != spec["trainer_sha256"]
    ):
        print(json.dumps(audit, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
