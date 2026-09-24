#!/usr/bin/env python3
"""Verify gsplat spherical_harmonics correct usage for this env."""
import torch
import sys, os
sys.path.insert(0, "experiments/r3")

from gsplat.cuda._wrapper import spherical_harmonics, fully_fused_projection

device = "cuda"
N = 128
W = H = 128

xyz = torch.randn(N, 3, device=device)
quats = torch.randn(N, 4, device=device)
scales = torch.randn(N, 3, device=device).exp() * 0.05
opac = torch.rand(N, device=device).sigmoid()
shs = torch.zeros(N, 16, 3, device=device)
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], device=device).unsqueeze(0)

# Correct usage based on gsplat 1.4 signatures
radii, means2d, depths, conics, comp = fully_fused_projection(
    xyz, None, quats, scales, viewmat, K, W, H, eps2d=0.1
)
print("projection OK")

# batch [1, N, ...]
dirs_batch = xyz.unsqueeze(0) - viewmat[:, :3, 3].unsqueeze(1)
colors_batch = spherical_harmonics(3, dirs_batch, shs.unsqueeze(0))
print("spherical_harmonics batch OK:", tuple(colors_batch.shape))

# no batch [N, ...]
dirs_flat = xyz - viewmat[:, :3, 3].unsqueeze(1)[0]
colors_flat = spherical_harmonics(3, dirs_flat, shs)
print("spherical_harmonics flat OK:", tuple(colors_flat.shape))
