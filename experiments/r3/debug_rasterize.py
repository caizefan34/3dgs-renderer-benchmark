#!/usr/bin/env python3
"""Debug rasterize_to_pixels error in gsplat env."""
import torch
import traceback

from gsplat.cuda._wrapper import (
    fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels,
)

N = 128
W, H = 128, 128
device = torch.device("cuda")

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
print("projection OK")

tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d, radii, depths, 16,
    (W + 15) // 16, (H + 15) // 16, sort=True
)
print("isect OK")

isect_offsets = isect_offset_encode(isect_ids, 1, (W + 15) // 16, (H + 15) // 16)
print("offsets OK")

# check shapes
print(f"means2d: {tuple(means2d.shape)} | conics: {tuple(conics.shape)}")
print(f"shs: {tuple(shs.shape)} | opacities: {tuple(opac.shape)}")
print(f"flatten_ids: {tuple(flatten_ids.shape)}")

try:
    render, render_alive = rasterize_to_pixels(
        means2d.unsqueeze(0), conics.unsqueeze(0), shs.unsqueeze(0), opac.unsqueeze(0),
        W, H, 16, isect_offsets, flatten_ids,
        backgrounds=None, masks=None, packed=False, absgrad=True,
    )
    print("rasterize OK", tuple(render.shape))
except Exception:
    traceback.print_exc()
