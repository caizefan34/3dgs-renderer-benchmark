#!/usr/bin/env python3
"""Minimal gsplat CUDA kernel execution test with correct signatures."""
import torch
import sys, os

print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
device = torch.device("cuda")

from gsplat.cuda._wrapper import (
    fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels,
)

N = 128
W, H = 128, 128
tile_size = 16

# Test each kernel
xyz = torch.randn(N, 3, device=device) * 0.1
quats = torch.randn(N, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.randn(N, 3, device=device).exp() * 0.05
opac = torch.rand(N, device=device).sigmoid()
shs = torch.zeros(N, 16, 3, device=device)
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], device=device).unsqueeze(0)

radii, means2d, depths, conics, compensation = fully_fused_projection(
    xyz, None, quats, scales, viewmat, K, W, H, eps2d=0.1
)
print("fully_fused_projection OK")
print(f"  radii: {tuple(radii.shape)} | means2d: {tuple(means2d.shape)} | depths: {tuple(depths.shape)}")

tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d, radii, depths, tile_size,
    (W + tile_size - 1) // tile_size, (H + tile_size - 1) // tile_size,
    sort=True,
)
print("isect_tiles OK")
print(f"  flatten_ids: {tuple(flatten_ids.shape)}")

isect_offsets = isect_offset_encode(isect_ids, 1, (W + tile_size - 1) // tile_size, (H + tile_size - 1) // tile_size)
print("isect_offset_encode OK")

render, render_alive = rasterize_to_pixels(
    means2d, conics, shs, opac.unsqueeze(0),
    W, H, tile_size, isect_offsets, flatten_ids,
    backgrounds=None, masks=None, packed=False, absgrad=True,
)
print("rasterize_to_pixels OK")
print(f"  render: {tuple(render.shape)} | alive: {tuple(render_alive.shape)}")

torch.cuda.synchronize()
print("\nALL KERNELS EXECUTE SUCCESSFULLY")

# Check camera sequence files
print("\n=== Camera sequence search ===")
for root, dirs, files in os.walk("data"):
    for f in files:
        if "camera" in f.lower() or "pose" in f.lower():
            print(f"  {os.path.join(root, f)}")
for root, dirs, files in os.walk("results"):
    for f in files:
        if "camera" in f.lower():
            print(f"  {os.path.join(root, f)}")
