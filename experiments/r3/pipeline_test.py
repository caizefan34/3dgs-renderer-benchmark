#!/usr/bin/env python3
"""Complete gsplat CUDA pipeline test with correct shapes (no batch dims)."""
import torch

from gsplat.cuda._wrapper import (
    fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels,
)

N = 128
W = H = 128
device = "cuda"

xyz = torch.randn(N, 3, device=device) * 0.1
quats = torch.randn(N, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.randn(N, 3, device=device).exp() * 0.05
opac = torch.rand(N, device=device).sigmoid()
shs = torch.randn(N, 16, 3, device=device) * 0.01
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

# 1. projection (per docs: returns means2d [N,2], depths [N,1], radii [N,1], conics [N,3], compensations [N,1])
radii, means2d, depths, conics, compensations = fully_fused_projection(
    xyz, None, quats, scales, viewmat, K, W, H, eps2d=0.1
)
print("1. fully_fused_projection OK")
print(f"   radii={tuple(radii.shape)} means2d={tuple(means2d.shape)} depths={tuple(depths.shape)} conics={tuple(conics.shape)}")

# 2. isect tiles
tile_size = 16
tile_w, tile_h = (W + tile_size - 1) // tile_size, (H + tile_size - 1) // tile_size
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d, radii, depths, tile_size, tile_w, tile_h, sort=True
)
print("2. isect_tiles OK")
print(f"   flatten_ids={tuple(flatten_ids.shape)}")

# 3. offset encode
isect_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h)
print("3. isect_offset_encode OK")

# 4. rasterize to pixels — colors must be [N, C] per v1.4 signature
colors = torch.randn(N, 3, device=device) * 0.01
render, render_alive = rasterize_to_pixels(
    means2d, conics, colors, opac,
    W, H, tile_size, isect_offsets, flatten_ids,
    backgrounds=None, masks=None, packed=False, abs_grad=True,
)
print("4. rasterize_to_pixels OK")
print(f"   render={tuple(render.shape)} alive={tuple(render_alive.shape)}")
print("\n*** FULL CUDA PIPELINE WORKS ***")
