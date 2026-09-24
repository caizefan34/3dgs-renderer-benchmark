#!/usr/bin/env python3
import json

# Check s23 provenance
path = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s23/garden/s0.750_30k.json"
d = json.load(open(path))

# Print all top-level string/dict fields
for k, v in d.items():
    if isinstance(v, str):
        print(f"{k}: {v}")
    elif isinstance(v, dict) and k in ("config", "provenance", "meta"):
        print(f"{k}: {json.dumps(v, indent=2)[:500]}")
    elif isinstance(v, (int, float)):
        print(f"{k}: {v}")

# Check if there's a separate provenance file
import os
s23_dir = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s23/garden"
for f in os.listdir(s23_dir):
    print(f"  file: {f}")
