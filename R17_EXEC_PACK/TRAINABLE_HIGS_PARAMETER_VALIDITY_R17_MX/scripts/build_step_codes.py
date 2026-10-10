#!/usr/bin/env python3
"""Build logs/step_exit_codes.json from logs/step_exit_codes.txt (runner bookkeeping)."""
import json
import sys
from pathlib import Path

logs = Path(sys.argv[1] if len(sys.argv) > 1 else "logs")
txt = logs / "step_exit_codes.txt"
codes = {}
if txt.is_file():
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            codes[parts[0]] = int(parts[1])
(logs / "step_exit_codes.json").write_text(json.dumps(codes, indent=2))
print(json.dumps(codes, indent=2))
