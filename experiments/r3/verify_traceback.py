#!/usr/bin/env python3
"""Full traceback for spherical_harmonics failure."""
import os, sys, traceback
sys.path.insert(0, "experiments/r3")
import torch

try:
    from gsplat.cuda._wrapper import spherical_harmonics, fully_fused_projection
    device = "cuda"
    torch.manual_seed(12345)
    N = 64
    W = H = 32

    xyz = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.randn(N, 3, device=device).exp() * 0.05
    opac = torch.rand(N, device=device).exp()
    shs = torch.zeros(N, 16, 3, device=device)
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    K = torch.tensor([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], 
                     device=device, dtype=torch.float32).unsqueeze(0)

    radii, means2d, depths, conics, compensation = fully_fused_projection(
        xyz, None, quats, scales, viewmat, K, W, H, eps2d=0.1
    )

    dirs = xyz - viewmat[0, :3, 3]
    print("dirs:", tuple(dirs.shape))
    print("shs:", tuple(shs.shape))
    colors = spherical_harmonics(3, dirs, shs)
    print("FLAT OK:", tuple(colors.shape))

    dirs_b = xyz.unsqueeze(0) - viewmat[:, :3, 3].unsqueeze(1)
    shs_b = shs.unsqueeze(0)
    print("dirs_b:", tuple(dirs_b.shape))
    print("shs_b:", tuple(shs_b.shape))
    colors_b = spherical_harmonics(3, dirs_b, shs_b)
    print("BATCH OK:", tuple(colors_b.shape))
except Exception:
    traceback.print_exc()
