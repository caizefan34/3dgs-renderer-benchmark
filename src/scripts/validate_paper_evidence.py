#!/usr/bin/env python
"""Validate submission claims against immutable, Git-tracked evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paper_evidence import PaperEvidenceError, validate_paper_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "paper" / "claims.json"))
    args = parser.parse_args()
    try:
        summary = validate_paper_evidence(args.manifest, repository_root=ROOT)
    except PaperEvidenceError as exc:
        print(f"paper evidence invalid: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
