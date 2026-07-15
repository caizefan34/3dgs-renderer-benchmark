#!/usr/bin/env python
"""Validate hashes for one or all official benchmark-suite scenes."""
import argparse
import os
import sys


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from benchmark_suite import load_benchmark_suite, resolve_suite_case  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["garden", "bicycle", "room"])
    parser.add_argument("--resolution", choices=["720p", "1080p", "4k"], default="1080p")
    args = parser.parse_args()

    suite = load_benchmark_suite()
    scene_ids = [args.scene] if args.scene else [scene["scene_id"] for scene in suite["scenes"]]
    for scene_id in scene_ids:
        case = resolve_suite_case(scene_id, args.resolution)
        print(f"validated {case['suite_case_id']}")


if __name__ == "__main__":
    main()
