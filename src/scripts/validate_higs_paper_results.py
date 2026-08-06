#!/usr/bin/env python
"""Validate HiGS full-training result JSON files against the paper protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_paper_results import HigsPaperResultError, validate_result_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark" / "higs-paper-protocol.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        report = validate_result_set(results, protocol, require_complete=args.require_complete)
    except (OSError, json.JSONDecodeError, HigsPaperResultError) as exc:
        print(f"HiGS paper results invalid: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
