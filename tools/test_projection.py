#!/usr/bin/env python3
"""Quick test: can gsplat projection generate real isect_ids?"""

import torch, math, time, sys
sys.stdout.reconfigure(line_buffering=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# Create a simple scene
n = 50000
h, w = 1080, 1920

# Random Gaussians in view frustum
fov_x = math.radians(50)
fov_y = math.radians(50)
tan_fov_x = math.tan(fov_x / 2)
tan_fov_y = math.tan(fov_y / 2)

depths = torch.exp(torch.empty(n).uniform_(math.log(0.5), math.log(50.0))).to(DEVICE)

# Random positions - uniformly in image plane, projected to 3D
uv_x = torch.rand(n, device=DEVICE) * 2 - 1
uv_y = torch.rand(n, device=DEVICE) * 2 - 1

means = torch.stack([uv_x * depths * tan_fov_x, uv_y * depths * tan_fov_y, -depths], dim=1)  # [n, 3]
means = means.unsqueeze(0)  # [1, n, 3]

quats = torch.randn(n, 4, device=DEVICE)
quats = (quats / quats.norm(dim=1, keepdim=True)).unsqueeze(0)

scales = (torch.rand(n, 3, device=DEVICE) * 0.02 + 0.005).unsqueeze(0)

opacities = torch.sigmoid(torch.randn(n, device=DEVICE)).unsqueeze(0)

colors = torch.rand(n, 3, device=DEVICE).unsqueeze(0)

# Camera
viewmat = torch.eye(4, device=DEVICE).unsqueeze(0)

fx = w / (2 * tan_fov_x)
fy = h / (2 * tan_fov_y)
K = torch.tensor([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], device=DEVICE, dtype=torch.float32).unsqueeze(0)

img_size = torch.tensor([h, w], device=DEVICE)

print(f"Scene tensors created. n={n}, means.shape={means.shape}", flush=True)

from gsplat import isect_tiles

print("Calling isect_tiles...", flush=True)
t0 = time.time()
means2d, radii, isect_ids, flatten_ids, n_isects = isect_tiles(
    means=means, quats=quats, scales=scales, opacities=opacities,
    viewmat=viewmat, K=K, img_size=img_size, tile_size=16, packed=True,
)
t1 = time.time()
print(f"isect_tiles done: n_isects={n_isects}, time={t1-t0:.3f}s", flush=True)
print(f"isect_ids.shape={isect_ids.shape}, dtype={isect_ids.dtype}", flush=True)
print(f"flatten_ids.shape={flatten_ids.shape}", flush=True)
print(f"means2d.shape={means2d.shape}, radii.shape={radii.shape}", flush=True)

# Check isect_ids
ids = isect_ids.cpu().numpy().astype('int64')
print(f"First 5 isect_ids: {ids[:5]}")
print(f"First 5 in hex:    {[hex(x) for x in ids[:5]]}")
print(f"isect_ids stats: min={ids.min()}, max={ids.max()}", flush=True)

# Check depths
depths_from_proj = -means[0, :, 2]
print(f"Depths: min={depths_from_proj.min().item():.4f}, max={depths_from_proj.max().item():.4f}", flush=True)

print("SUCCESS", flush=True)
