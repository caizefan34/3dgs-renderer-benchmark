#!/usr/bin/env python
"""Validate JSON artifacts against repository schemas."""
import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from schema_validation import validate_json_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--json", nargs="+", required=True)
    args = parser.parse_args()

    schema_path = args.schema
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(REPO_ROOT, schema_path)
    for json_path in args.json:
        candidate = json_path if os.path.isabs(json_path) else os.path.join(REPO_ROOT, json_path)
        validate_json_file(candidate, schema_path)
        print(f"validated {json_path}")


if __name__ == "__main__":
    main()

