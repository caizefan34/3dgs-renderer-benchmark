#!/usr/bin/env python3
"""Check what keys load_ply returns."""
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from benchmark_framework import load_ply
import inspect
src = inspect.getsource(load_ply)
print(src[:2000])
print("---")
# Also check what keys it returns
from pathlib import Path
ply_path = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/data/official/mipnerf360/room/point_cloud.ply")
data = load_ply(str(ply_path), device="cpu")
print(f"Keys: {list(data.keys())}")
for k, v in data.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape} {v.dtype}")
    else:
        print(f"  {k}: {type(v)} = {v}")
