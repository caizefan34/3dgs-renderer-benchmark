#!/usr/bin/env python3
"""Verify correct gsplat spherical_harmonics shape usage."""
import torch

from gsplat.cuda._wrapper import spherical_harmonics

device = "cuda"
N = 64
dirs = torch.randn(1, N, 3, device=device)       # [1, N, 3]
coeffs_batched = torch.zeros(1, N, 16, 3, device=device)  # [1, N, K, D]

try:
    colors = spherical_harmonics(3, dirs, coeffs_batched)
    print("BATCHED coeffs OK:", tuple(colors.shape))
except Exception as e:
    print("BATCHED coeffs FAILED:", type(e).__name__, str(e)[:300])

coeffs_flat = coeffs_batched[0]  # [N, K, D]
try:
    colors = spherical_harmonics(3, dirs, coeffs_flat)
    print("FLAT coeffs OK:", tuple(colors.shape))
except Exception as e:
    print("FLAT coeffs FAILED:", type(e).__name__, str(e)[:300])

try:
    dirs_flat = dirs[0]  # [N, 3]
    colors = spherical_harmonics(3, dirs_flat, coeffs_flat)
    print("FLAT dirs+coeffs OK:", tuple(colors.shape))
except Exception as e:
    print("FLAT dirs+coeffs FAILED:", type(e).__name__, str(e)[:300])
