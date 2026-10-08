#!/usr/bin/env python3
"""
R5 Correctness Validation - Compare Frozen HiGS vs Patched HiGS
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

print("=== R5 Correctness Validation ===\n")

# Load checkpoint
checkpoint_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
print(f"Loading checkpoint: {checkpoint_path}")

# Sample 580K Gaussians for correctness test
checkpoint = torch.load(checkpoint_path, map_location='cpu')
N_full = checkpoint['xyz'].shape[0]
N_test = 580_000

# Deterministic sampling
np.random.seed(42)
indices = np.sort(np.random.choice(N_full, N_test, replace=False))

print(f"Full N: {N_full:,}")
print(f"Test N: {N_test:,}")
print(f"Sampling: deterministic (seed=42)\n")

# Create test states
test_states = {
    "580K": indices[:580_000],
    "2.0M": indices[:2_000_000] if len(indices) >= 2_000_000 else np.sort(np.random.choice(N_full, 2_000_000, replace=False))
}

print("Test states prepared:")
for name, idx in test_states.items():
    print(f"  {name}: {len(idx):,} Gaussians")

print("\nCorrectness criteria:")
print("  - RGB output difference: < 1e-5")
print("  - Alpha difference: < 1e-5")
print("  - Depth difference: < 1e-5")
print("  - Gradient finite: all True")
print("  - Repeatability: stable across 3 runs")

print("\nNote: Full validation requires building patched HiGS.")
print("This script prepares test infrastructure.")
