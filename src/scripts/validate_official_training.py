#!/usr/bin/env python
"""Validate official-dataset training policy and renderer registry."""
import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from datasets.official import load_official_dataset_manifest, validate_training_manifest  # noqa: E402
from renderers.candidate_registry import load_renderer_candidates, validate_renderer_candidates  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=os.path.join(REPO_ROOT, "data", "datasets", "official_training_datasets.json"))
    parser.add_argument("--renderers", default=os.path.join(REPO_ROOT, "data", "renderers", "renderer_candidates.json"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "datasets": validate_training_manifest(load_official_dataset_manifest(args.datasets)),
        "renderers": validate_renderer_candidates(load_renderer_candidates(args.renderers)),
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

