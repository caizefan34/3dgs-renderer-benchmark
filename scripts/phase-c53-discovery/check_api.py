#!/usr/bin/env python3
"""Check what meta contains from rasterization."""
import inspect
from gsplat import rasterization

src = inspect.getsource(rasterization)
# Find where meta is constructed
for i, line in enumerate(src.split('\n')):
    if 'meta' in line.lower() and ('=' in line or 'dict' in line.lower() or 'return' in line.lower()):
        print(f"{i}: {line.rstrip()}")
print("\n--- Full return section ---")
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'meta' in line:
        start = max(0, i-3)
        end = min(len(lines), i+3)
        for j in range(start, end):
            print(f"{j}: {lines[j].rstrip()}")
        print("...")
