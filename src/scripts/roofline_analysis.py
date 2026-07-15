#!/usr/bin/env python
"""Classify a workload using optional roofline-style counters."""
import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.roofline import roofline_classification  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arithmetic-intensity", type=float, required=True)
    parser.add_argument("--memory-bandwidth-utilization", type=float, required=True)
    parser.add_argument("--compute-utilization", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = roofline_classification(
        args.arithmetic_intensity,
        args.memory_bandwidth_utilization,
        args.compute_utilization,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()

