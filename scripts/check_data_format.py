#!/usr/bin/env python3
"""Check PLY and camera data format."""
import sys
sys.path.insert(0, '/mnt/storage_pool/3dgs-renderer-benchmark/repo/src')
sys.path.insert(0, '/mnt/storage_pool/3dgs-renderer-benchmark/repo')
from benchmark_framework import load_ply, load_cameras_from_json
import torch

# Check PLY
state = load_ply('/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/point_cloud.ply', device='cpu')
print('PLY keys:', list(state.keys()))
for k, v in state.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.shape} {v.dtype}')
    else:
        print(f'  {k}: {v}')

# Check cameras
cams = load_cameras_from_json('/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json', device='cpu')
print(f'\nCameras: {len(cams)}')
c = cams[0]
print('Camera attrs:', [a for a in dir(c) if not a.startswith('_')])
print('width:', c.width, 'height:', c.height)
print('image_name:', getattr(c, 'image_name', 'N/A'))
print('fx:', getattr(c, 'fx', 'N/A'), 'fy:', getattr(c, 'fy', 'N/A'))
