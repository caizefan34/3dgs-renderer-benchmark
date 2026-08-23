#!/usr/bin/env python3
"""Final test: JIT compile gsplat in VS dev command prompt env."""
import os, sys

# Strip any trailing spaces from CUDA_PATH
if "CUDA_PATH" in os.environ:
    os.environ["CUDA_PATH"] = os.environ["CUDA_PATH"].strip()

import torch
import torch.utils.cpp_extension as cpp_ext

# Fix CJK code page issue
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')

from gsplat import rasterization
print('Import OK', flush=True)

device = 'cuda'
means = torch.randn(100, 3, device=device)
quats = torch.nn.functional.normalize(torch.randn(100, 4, device=device), dim=-1)
scales = torch.rand(100, 3, device=device) * 0.1
opacities = torch.sigmoid(torch.randn(100, device=device))
colors = torch.randn(100, 16, 3, device=device)
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[500, 0, 960], [0, 500, 540], [0, 0, 1]], device=device).unsqueeze(0)

rendered, alpha, info = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities,
    colors=colors, viewmats=viewmat, Ks=K, width=1920, height=1080,
    tile_size=16, sh_degree=3, packed=True, render_mode='RGB'
)
print(f'Rendered: {rendered.shape}', flush=True)
print('SUCCESS', flush=True)
