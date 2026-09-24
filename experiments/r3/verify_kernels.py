#!/usr/bin/env python3
"""Verify gsplat CUDA kernels actually execute + find camera sequence sources."""
import os, sys, glob, json
import torch

print(f"torch: {torch.__version__} | cuda avail: {torch.cuda.is_available()}")

# 1. Test gsplat kernel execution with tiny tensors (no JIT needed if prebuilt)
try:
    import gsplat
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels,
    )
    print("wrapper import: OK")

    # minimal real pipeline test
    N = 64
    device = torch.device("cuda")
    means3d = torch.randn(1, N, 3, device=device) * 0.1
    scales = torch.randn(1, N, 3, device=device).exp() * 0.05
    quats = torch.randn(1, N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities = torch.rand(N, device=device).sigmoid()
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    K = torch.tensor([[128, 0, 64], [0, 128, 64], [0, 0, 1]], device=device).unsqueeze(0)

    radii, means2d, depths, conics, compensation = fully_fused_projection(
        means3d, None, quats, scales, viewmat, K, 128, 128
    )
    print("fully_fused_projection: OK | radii:", tuple(radii.shape))

    # isect
    tile_size = 16
    tile_w, tile_h = 8, 8
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_w, tile_h, sort=True
    )
    print("isect_tiles: OK | flatten_ids:", tuple(flatten_ids.shape))

except Exception as e:
    import traceback
    print(f"gsplat kernel FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. Camera sequence — search for any existing generator or camera data
print("\n=== Camera sequence data ===")
for pat in [
    "data/camera_sequence*", "data/*camera*", "data/camera_presets/*",
    "experiments/r3/*.py", "scripts/**/*camera*", "src/**/*camera*"
]:
    for f in glob.glob(pat, recursive=True):
        sz = os.path.getsize(f)
        print(f"  {f} ({sz/1024:.1f} KB)" if sz > 0 else f"  {f} (empty)")
