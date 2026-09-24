#!/usr/bin/env python3
"""Check if tile32 reduces VRAM enough for 500-step training on bicycle."""
import sys, torch
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gsplat import rasterization
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
import numpy as np
from PIL import Image

DEVICE = "cuda"
W, H = 1920, 1080

torch.cuda.reset_peak_memory_stats()

sfm = load_ply(str(REPO_ROOT / "data/official/mipnerf360/bicycle/point_cloud.ply"), device=DEVICE)
N = sfm["xyz"].shape[0]
xyz = sfm["xyz"].detach().clone().requires_grad_(True)
rots = torch.nn.functional.normalize(sfm["rotations"], dim=-1).contiguous().requires_grad_(True)
scales = torch.exp(sfm["scales"]).contiguous().requires_grad_(True)
opacity = torch.sigmoid(sfm["opacity"]).contiguous().requires_grad_(True)
shs = sfm["shs"].detach().clone().requires_grad_(True)

# Load first real camera
real_cameras = load_cameras_from_json(str(REPO_ROOT / "data/official/mipnerf360/bicycle/cameras.json"), device="cpu")
real_cameras = resize_cameras(real_cameras, W, H)
cam0 = real_cameras[0]
for attr in ["viewmatrix", "projmatrix", "camera_center", "world_view_transform", "full_proj_transform", "K"]:
    t = getattr(cam0, attr)
    if isinstance(t, torch.Tensor):
        setattr(cam0, attr, t.to(DEVICE))
rv = cam0.viewmatrix.unsqueeze(0)
rK = cam0.K.unsqueeze(0)

gt_files = sorted([f for f in (REPO_ROOT / "data/datasets/mipnerf360/bicycle/images").iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
with Image.open(gt_files[0]) as source:
    if source.width != W or source.height != H:
        source = source.resize((W, H), Image.LANCZOS)
    rgba = torch.from_numpy(np.array(source.convert("RGBA"), dtype=np.uint8))
gt = rgba.to(DEVICE, dtype=torch.float32)[..., :3] / 255.0

for tile_size in [16, 32]:
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam([
        {"params": [xyz], "lr": 1.6e-4 * 46.64, "eps": 1e-15},
        {"params": [rots], "lr": 1e-3, "eps": 1e-15},
        {"params": [scales], "lr": 5e-3, "eps": 1e-15},
        {"params": [opacity], "lr": 5e-2, "eps": 1e-15},
        {"params": [shs], "lr": 2.5e-3, "eps": 1e-15},
    ])

    for step in range(5):
        optimizer.zero_grad()
        rendered, alpha, meta = rasterization(
            means=xyz, quats=rots, scales=scales, opacities=opacity, colors=shs,
            viewmats=rv, Ks=rK, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=3,
        )
        rendered = rendered[0].clamp(0, 1)
        loss = ((rendered - gt) ** 2).mean()
        loss.backward()
        optimizer.step()

    peak = torch.cuda.max_memory_allocated()/1024**3
    free = torch.cuda.mem_get_info()[0]/1024**3
    print(f"tile_size={tile_size}: peak={peak:.3f}GB, free={free:.3f}GB")
