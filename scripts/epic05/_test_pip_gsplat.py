"""Test the pip-installed gsplat's csrc tile size support."""
import sys

# Remove dev source paths to use pip-installed gsplat
sys.path = [p for p in sys.path 
    if 'artifacts' not in p and 'renderer-sources' not in p and 'task_b_gsplat' not in p]

import gsplat
print("gsplat:", gsplat.__file__)
print("ver:", gsplat.__version__)

# Check if csrc is loaded
try:
    print("csrc:", gsplat.csrc)
except AttributeError:
    print("No csrc module available")

# Try importing rasterization
from gsplat import rasterization
print("rasterization imported OK")

# Test each tile size
import torch
device = 'cuda'
means = torch.randn(100, 3, device=device)
quats = torch.nn.functional.normalize(torch.randn(100, 4, device=device), dim=-1)
scales = torch.rand(100, 3, device=device) * 0.1
opacities = torch.sigmoid(torch.randn(100, device=device))
colors = torch.randn(100, 16, 3, device=device)
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[500, 0, 960], [0, 500, 540], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

for ts in [4, 8, 16, 32]:
    try:
        rendered, alpha, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmat, Ks=K, width=1920, height=1080,
            tile_size=ts, sh_degree=3, packed=True, render_mode='RGB'
        )
        print(f"  tile_size={ts}: OK, shape={rendered.shape}")
    except Exception as e:
        err = str(e).strip()[:100]
        print(f"  tile_size={ts}: FAILED - {err}")
