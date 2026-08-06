#!/usr/bin/env python
"""Validate the survey, HiGS, and compression paper tracks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research_program import ResearchProgramError, validate_research_program  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", dest="manifests")
    args = parser.parse_args()
    manifests = args.manifests or [
        ROOT / "paper" / "survey-claims.json",
        ROOT / "paper" / "higs-claims.json",
        ROOT / "paper" / "compression-claims.json",
    ]
    try:
        summary = validate_research_program(manifests, repository_root=ROOT)
    except ResearchProgramError as exc:
        print(f"research program invalid: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
