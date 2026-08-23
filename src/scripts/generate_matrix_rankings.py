#!/usr/bin/env python
"""Generate tier-separated Benchmark Matrix v2 rankings."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_matrix import generate_matrix_report, load_results, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--suite", default=str(ROOT / "benchmark" / "suite.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "generated"))
    args = parser.parse_args()

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    documents, rejected_files = load_results(args.inputs)
    report = generate_matrix_report(documents, suite)
    report["rejected_files"] = rejected_files
    write_report(report, args.output_dir)
    print(f"wrote {Path(args.output_dir) / 'ranking.json'}")
    print(f"wrote {Path(args.output_dir) / 'ranking.md'}")
    return 1 if rejected_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
