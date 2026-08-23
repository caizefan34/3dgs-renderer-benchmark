#!/usr/bin/env python
"""Validate local links in public repository entry points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from public_links import PublicLinkError, validate_public_links  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry-points",
        nargs="+",
        default=["README.md", "docs/README.md", "paper/README.md", "CONTRIBUTING.md"],
    )
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args()
    try:
        summary = validate_public_links(ROOT, args.entry_points, args.require_tracked)
    except PublicLinkError as exc:
        print(f"public link validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
