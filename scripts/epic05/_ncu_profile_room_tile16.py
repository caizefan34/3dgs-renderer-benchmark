#!/usr/bin/env python3
"""Standalone ncu profile target: room tile16 rasterize kernel."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from gsplat import rasterization
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
scene = load_ply("data/official/mipnerf360/room/point_cloud.ply", device="cuda")
cams = resize_cameras(load_cameras_from_json("data/official/mipnerf360/room/cameras.json", device="cuda"), 1920, 1080)
cam = cams[0]
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales = torch.exp(scene["scales"]).contiguous()
ops = torch.sigmoid(scene["opacity"]).squeeze(-1).contiguous()
shs = scene["shs"].contiguous()
xyz = scene["xyz"].contiguous()
# Warmup
for _ in range(3):
    rendered, _, _ = rasterization(means=xyz, quats=quats, scales=scales, opacities=ops, colors=shs, viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height, tile_size=16, sh_degree=3, packed=True, render_mode="RGB")
    torch.cuda.synchronize()
# Measure
for _ in range(5):
    rendered, _, _ = rasterization(means=xyz, quats=quats, scales=scales, opacities=ops, colors=shs, viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height, tile_size=16, sh_degree=3, packed=True, render_mode="RGB")
    torch.cuda.synchronize()
print("PROFILING COMPLETE")
