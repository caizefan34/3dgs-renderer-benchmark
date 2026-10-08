#!/usr/bin/env python3
"""
Test HiGS return structure
"""
import sys
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
import torch
import torch.nn.functional as F
from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

# Create dummy gaussians
N = 10000
means = torch.randn(N, 3)
quats = F.normalize(torch.randn(N, 4), dim=-1)
scales = torch.exp(torch.randn(N, 3) * 0.1)
opacities = torch.sigmoid(torch.randn(N))
shs = torch.randn(N, 16, 3)

# Camera
width, height = 960, 540
K = torch.tensor([
    [width, 0, width/2],
    [0, height, height/2],
    [0, 0, 1]
], dtype=torch.float32)

viewmat = torch.eye(4)
viewmat[2, 3] = 3.0

result = rasterize_gaussian_higs_frozen(
    means, quats, scales, opacities, shs,
    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
    width=width, height=height,
    sh_degree=3,
    enable_culling=True,
    use_higs_culling=True
)

print("Result type:", type(result))
print("Result keys:", result.keys() if isinstance(result, dict) else "Not a dict")
print("\nResult contents:")
for k, v in result.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: {type(v)}")
