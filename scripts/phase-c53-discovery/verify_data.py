#!/usr/bin/env python3
"""Verify the collected raw data structure."""
import numpy as np
from pathlib import Path

filepath = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery/room_raw.npz")
data = np.load(filepath, allow_pickle=True)
print("Keys in raw data:")
for key in sorted(data.keys()):
    val = data[key]
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

# Check signal values
print("\n--- Signal statistics (checkpoint 500) ---")
for sig in ["prev_grad_norm", "ema_grad_norm", "visibility_count", "screen_radius_mean",
            "opacity", "scale_norm", "age"]:
    key = f"cp500_s_{sig}"
    if key in data:
        vals = data[key]
        print(f"  {sig}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}, "
              f"nonzero={np.count_nonzero(vals)}/{len(vals)}")

# Check future utility
print("\n--- Future utility (checkpoint 500, Δ=10) ---")
for key in sorted(data.keys()):
    if key.startswith("cp500_d10_"):
        vals = data[key]
        print(f"  {key.replace('cp500_d10_', '')}: mean={vals.mean():.4f}, "
              f"std={vals.std():.4f}, nonzero={np.count_nonzero(vals)}/{len(vals)}")

print("\n--- Densification outcome ---")
if "cp500_outcome" in data:
    outcome = data["cp500_outcome"]
    print(f"  0=unchanged: {(outcome==0).sum()}")
    print(f"  1=cloned: {(outcome==1).sum()}")
    print(f"  2=split: {(outcome==2).sum()}")
    print(f"  3=pruned: {(outcome==3).sum()}")
