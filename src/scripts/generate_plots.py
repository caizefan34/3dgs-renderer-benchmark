#!/usr/bin/env python
"""Generate publication-oriented benchmark plots."""
import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.plots import generate_plots  # noqa: E402
from leaderboard.generator import load_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outputs = generate_plots(load_records(args.inputs), args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

