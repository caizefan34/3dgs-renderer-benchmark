#!/usr/bin/env python3
"""Inspect the full nested structure of a result JSON."""
import json, glob

files = sorted(glob.glob(r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results\*.json"))
seen = set()
for f in files:
    d = json.load(open(f))
    key = (d.get("method"), d.get("scene"))
    if key in seen:
        continue
    seen.add(key)
    print(f"--- {f.split(chr(92))[-1]} ---")
    print(json.dumps(d, indent=2)[:4000])
    print()
    if len(seen) >= 2:
        break
