#!/usr/bin/env python
"""Wait until a JSON session reaches an expected status."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def wait_for_status(path: Path, expected: str, poll_seconds: float) -> dict:
    while True:
        if path.is_file():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                document = None
            if document is not None and document.get("status") == expected:
                return document
        time.sleep(poll_seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    document = wait_for_status(args.path.resolve(), args.status, args.poll_seconds)
    print(json.dumps({"path": str(args.path.resolve()), "status": document["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
