#!/usr/bin/env python
"""Generate leaderboard.json, leaderboard.md, and leaderboard.html."""
import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from leaderboard import generate_leaderboard, load_records, write_leaderboard  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    records = load_records(args.inputs)
    leaderboard = generate_leaderboard(records)
    write_leaderboard(leaderboard, args.output_dir)


if __name__ == "__main__":
    main()

