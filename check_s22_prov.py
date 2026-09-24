#!/usr/bin/env python3
import json

# Check s22 garden baseline_30k.json for provenance
path = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/garden/baseline_30k.json"
d = json.load(open(path))
print("Keys:", list(d.keys()))
# Look for provenance/trainer info
for k in d:
    if k in ["provenance", "config", "trainer", "script", "meta"]:
        print(f"\n{k}: {json.dumps(d[k], indent=2)[:500]}")
# Also check top-level fields
for k, v in d.items():
    if isinstance(v, str) and ("script" in k.lower() or "trainer" in k.lower() or "source" in k.lower()):
        print(f"\n{k}: {v}")
    if isinstance(v, dict) and len(v) < 10:
        print(f"\n{k}: {json.dumps(v, indent=2)[:300]}")
