#!/usr/bin/env python3
import numpy as np
from pathlib import Path

filepath = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery/room_raw.npz")
data = np.load(filepath, allow_pickle=True)
keys = sorted(data.keys())
print(f"Total keys: {len(keys)}")
print("\nFirst 30 keys:")
for k in keys[:30]:
    print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

print("\nKeys with 'd10':")
for k in keys:
    if "d10" in k:
        print(f"  {k}: shape={data[k].shape}")

print("\nKeys with 'cp500':")
for k in keys:
    if "cp500" in k:
        print(f"  {k}: shape={data[k].shape}")

print("\nKeys with 's_':")
for k in keys[:10]:
    if "_s_" in k:
        print(f"  {k}: shape={data[k].shape}")
