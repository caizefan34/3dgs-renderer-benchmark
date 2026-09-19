#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1]))
for key in d:
    if isinstance(d[key], dict) and "passed" in d[key]:
        r = d[key]
        print(f"\n=== {key} ===")
        print(f"  passed: {r['passed']}")
        if "comparison_vs_baseline" in r:
            for name, m in r["comparison_vs_baseline"].items():
                print(f"  {name}: max_abs={m['max_abs']:.6e} nan_or_inf={m['nan_or_inf']}")
        if "stale_row0_zero" in r:
            for name, m in r["stale_row0_zero"].items():
                print(f"  stale {name}: is_zero={m['is_zero']} max_abs={m['max_abs']:.6e}")
        if "stale_row_zero_b1" in r:
            for name, m in r["stale_row_zero_b1"].items():
                print(f"  stale {name}: is_zero={m['is_zero']} max_abs={m['max_abs']:.6e}")
print(f"\nOverall passed: {d.get('passed')}")
