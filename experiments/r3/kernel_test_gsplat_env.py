#!/usr/bin/env python3
"""Test gsplat kernels using the gsplat conda env."""
import sys, os
import torch

print(f"Python: {sys.executable}")
print(f"torch: {torch.__version__}")
print(f"cuda avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  cuda: {torch.version.cuda}")

try:
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels,
    )
    print("gsplat wrapper: OK")

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
    print("fully_fused_projection: OK | radii:", tuple(radii.shape))

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, 16,
        (W + 15) // 16, (H + 15) // 16, sort=True
    )
    print("isect_tiles: OK | flatten_ids:", tuple(flatten_ids.shape))

    isect_offsets = isect_offset_encode(isect_ids, 1, (W + 15) // 16, (H + 15) // 16)
    print("isect_offset_encode: OK")

    render, render_alive = rasterize_to_pixels(
        means2d, conics, shs, opac.unsqueeze(0),
        W, H, 16, isect_offsets, flatten_ids,
        backgrounds=None, masks=None, packed=False, absgrad=True,
    )
    print("rasterize_to_pixels: OK | render:", tuple(render.shape))
    torch.cuda.synchronize()
    print("\n*** ALL GSPlAT KERNELS EXECUTE SUCCESSFULLY ***")

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
