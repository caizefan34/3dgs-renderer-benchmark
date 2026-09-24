#!/usr/bin/env python3
"""Scan all result JSON files and print a compact summary of available fields."""
import json, glob, os, sys
from collections import defaultdict

ROOTS = [
    r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results",
    r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results",
]
for root in ROOTS:
    if not os.path.isdir(root):
        print(f"[missing] {root}")
        continue
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    print(f"\n### {root}: {len(files)} files")
    # group by workload
    wl = defaultdict(list)
    for f in files:
        base = os.path.basename(f)
        wl_name = base.split("--")[0]
        wl[wl_name].append(base)
    for k in sorted(wl):
        print(f"  {k}: {len(wl[k])}")
        for b in wl[k][:3]:
            print(f"    e.g. {b}")
    # dump top-level keys of first file
    if files:
        import json
        with open(files[0]) as fh:
            j = json.load(fh)
        print(f"  first-file keys: {sorted(j.keys())}")
        print(f"  performance keys: {sorted(j.get('performance', {}).keys()) if isinstance(j.get('performance'), dict) else j.get('performance')}")
        print(f"  quality keys: {sorted(j.get('quality', {}).keys()) if isinstance(j.get('quality'), dict) else j.get('quality')}")
        print(f"  resources keys: {sorted(j.get('resources', {}).keys()) if isinstance(j.get('resources'), dict) else j.get('resources')}")
