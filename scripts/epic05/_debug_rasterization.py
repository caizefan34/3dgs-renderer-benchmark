#!/usr/bin/env python3
"""Debug: test rasterization call."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import torch
import torch.nn.functional as F
from benchmark_framework.scene import load_ply
from gsplat import rasterization

DEVICE = torch.device("cuda:0")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ply_path = os.path.join(REPO_ROOT, "data", "official", "mipnerf360", "room", "point_cloud.ply")
scene = load_ply(ply_path, device=DEVICE)

import json as _json
cam_path = os.path.join(REPO_ROOT, "data", "official", "mipnerf360", "room", "cameras.json")
with open(cam_path) as f:
    cams = _json.load(f)
cam = cams[0]

W, H = 1920, 1080
fx = cam.get('fx', cam.get('fl_x', 500.0))
fy = cam.get('fy', cam.get('fl_y', 500.0))
cx = cam.get('cx', cam.get('pp_x', W/2))
cy = cam.get('cy', cam.get('pp_y', H/2))

K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=DEVICE).unsqueeze(0)

if 'transform_matrix' in cam:
    c2w = torch.tensor(cam['transform_matrix'], dtype=torch.float32, device=DEVICE)
else:
    c2w = torch.eye(4, dtype=torch.float32, device=DEVICE)

viewmat = torch.inverse(c2w).unsqueeze(0)

print(f"viewmat shape: {viewmat.shape}")
print(f"K shape: {K.shape}")

means = scene['xyz'].unsqueeze(0)
quats = F.normalize(scene['rotations'], dim=-1).unsqueeze(0)
scales = scene['scales'].unsqueeze(0)
opacities = torch.sigmoid(scene['opacity']).unsqueeze(0)
colors = scene['shs'].unsqueeze(0)

print(f"means: {means.shape}")
print(f"quats: {quats.shape}")
print(f"scales: {scales.shape}")
print(f"opacities: {opacities.shape}")
print(f"colors: {colors.shape}")
print(f"sh_degree: {scene['sh_degree']}")

# Try rasterization call
try:
    render_colors, render_alphas, meta = rasterization(
        means, quats, scales, opacities, colors,
        viewmat, K, W, H,
        tile_size=16, packed=True, sh_degree=scene['sh_degree'],
    )
    print(f"\nSUCCESS! render_colors: {render_colors.shape}")
    print(f"isect_ids: {meta['isect_ids'].shape}")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
