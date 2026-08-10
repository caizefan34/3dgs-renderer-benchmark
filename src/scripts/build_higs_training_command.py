#!/usr/bin/env python
"""Audit a gsplat source tree and emit one frozen paper-training invocation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import (  # noqa: E402
    HigsTrainingCommandError,
    build_training_invocation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmark" / "higs-paper-protocol.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_dir = args.source_dir or (
        ROOT
        / "artifacts"
        / "renderer-sources"
        / ("gsplat-official" if args.method == "gsplat" else "gsplat-higs")
    )
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        invocation = build_training_invocation(
            protocol=protocol,
            method=args.method,
            scene=args.scene,
            seed=args.seed,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            source_dir=source_dir,
            repository_root=ROOT,
            protocol_path=args.protocol,
        )
        payload = json.dumps(invocation, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    except (OSError, json.JSONDecodeError, HigsTrainingCommandError) as exc:
        print(f"HiGS training command blocked: {exc}", file=sys.stderr)
        return 1
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
