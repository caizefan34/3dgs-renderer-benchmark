#!/usr/bin/env python
"""Validate and optionally export the independent HiGS ablation protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_ablation_protocol import (  # noqa: E402
    HigsAblationProtocolError,
    build_ablation_experiment_plan,
    validate_ablation_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmark" / "higs-ablation-protocol.json",
    )
    parser.add_argument("--output-plan", type=Path)
    args = parser.parse_args()
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        report = validate_ablation_protocol(protocol)
        if args.output_plan:
            args.output_plan.parent.mkdir(parents=True, exist_ok=True)
            args.output_plan.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "jobs": build_ablation_experiment_plan(protocol),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    except (OSError, json.JSONDecodeError, HigsAblationProtocolError) as exc:
        print(f"HiGS ablation protocol invalid: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
