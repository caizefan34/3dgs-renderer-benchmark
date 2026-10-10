#!/usr/bin/env python3
"""Regenerate logs/step_exit_codes.json from logs/step_exit_codes.txt."""
import json
import sys
from pathlib import Path

txt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/step_exit_codes.txt")
codes = {}
for line in txt.read_text().splitlines():
    if " " in line:
        k, v = line.rsplit(" ", 1)
        try:
            codes[k] = int(v)
        except ValueError:
            pass
Path("logs/step_exit_codes.json").write_text(json.dumps(codes, indent=2))
print(json.dumps(codes, indent=2))
