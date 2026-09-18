#!/usr/bin/env python3
"""Check gsplat rasterization metadata keys."""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch, json
import numpy as np
from gsplat import rasterization

ckpt = torch.load('results/reference_v1/room_30k/checkpoints/iter_30000.pt', map_location='cuda')
xyz = ckpt['xyz'].cuda().float().requires_grad_(True)
rot = ckpt['rotation'].cuda().float().requires_grad_(True)
scl = ckpt['scaling'].cuda().float().requires_grad_(True)
opc = ckpt['opacity'].cuda().float().requires_grad_(True)
shs = ckpt['shs'].cuda().float().requires_grad_(True)

cams = json.load(open('data/official/mipnerf360/room/cameras.json'))
cam = cams[0]
w, h = cam['width'], cam['height']
fx, fy = cam['fx'], cam['fy']
cx, cy = w/2, h/2
viewmat = np.eye(4, dtype=np.float32)
viewmat[:3,:3] = np.array(cam['rotation'], dtype=np.float32)
viewmat[:3,3] = np.array(cam['position'], dtype=np.float32)
viewmat = torch.from_numpy(viewmat).cuda().float()
K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=torch.float32).cuda()
tw, th = 1920, (int(h*1920/w)//16)*16
sx, sy = tw/w, th/h
r, a, meta = rasterization(
    means=xyz, quats=rot, scales=scl, opacities=opc, colors=shs,
    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
    width=tw, height=th, tile_size=16, packed=False,
    sh_degree=3, radius_clip=0.0, eps2d=0.1,
    render_mode='RGB', absgrad=True
)
print('render:', r.shape, a.shape)
print('meta keys:', list(meta.keys()))
for k, v in meta.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.shape} {v.dtype}')
    else:
        print(f'  {k}: {type(v)}')
