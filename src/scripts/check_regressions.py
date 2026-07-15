#!/usr/bin/env python
"""Compare candidate benchmark JSON against a baseline."""
import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.regression import compare_regressions  # noqa: E402
from leaderboard.generator import load_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps-drop-pct", type=float, default=5.0)
    parser.add_argument("--latency-increase-pct", type=float, default=5.0)
    parser.add_argument("--vram-increase-pct", type=float, default=10.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    report = compare_regressions(
        load_records(args.baseline),
        load_records(args.candidate),
        {
            "fps_drop_pct": args.fps_drop_pct,
            "latency_increase_pct": args.latency_increase_pct,
            "vram_increase_pct": args.vram_increase_pct,
        },
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(report["status"])
    if args.fail_on_regression and report["status"] == "regression":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

