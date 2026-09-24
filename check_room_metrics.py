#!/usr/bin/env python3
import json

path = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/training_metrics.json"
d = json.load(open(path))
print("Keys:", list(d.keys()) if isinstance(d, dict) else f"list of {len(d)}")
if isinstance(d, dict):
    for k in list(d.keys())[:5]:
        v = d[k]
        print(f"  {k}: {type(v).__name__} = {v if not isinstance(v, (list, dict)) else f'len={len(v)}'}")
    # Check for eval points
    for key in ["eval_points", "metrics", "psnr_ssim", "evaluations"]:
        if key in d:
            print(f"\n{key}:")
            vals = d[key]
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, dict) and v.get("iter", 0) in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
                        print(f"  {v}")
            elif isinstance(vals, dict):
                for k2, v2 in list(vals.items())[:5]:
                    print(f"  {k2}: {v2}")
