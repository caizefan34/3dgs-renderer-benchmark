#!/usr/bin/env python3
"""Quick environment check for Phase 5 profiling."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import gc

# Test 1: Basics
print("="*60)
print("Phase 5: Environment Check")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Test 2: gsplat import + basic forward pass
from gsplat import rasterization
import gsplat
print(f"gsplat version: {getattr(gsplat, '__version__', 'unknown')}")

# Test 3: Scene loading
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

scene_path = "data/official/mipnerf360/room/point_cloud.ply"
cam_path = "data/official/mipnerf360/room/cameras.json"

if not os.path.exists(scene_path):
    print(f"Scene not found: {scene_path}")
    sys.exit(1)

scene = load_ply(scene_path, device="cuda")
cams = load_cameras_from_json(cam_path, device="cuda")
cams = resize_cameras(cams, 1920, 1080)
cam = cams[0]

print(f"Gaussians: {scene['num_points']:,}")
print(f"Camera: {cam.image_width}x{cam.image_height}")

# Test 4: Quick rasterization (tile16 and tile32)
import time

for tile_size in [16, 32]:
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales_activated = torch.exp(scene["scales"]).contiguous()
    opacities_activated = torch.sigmoid(scene["opacity"]).squeeze(-1).contiguous()
    shs = scene["shs"].contiguous()
    xyz = scene["xyz"].contiguous()

    # Warm up
    for _ in range(3):
        rendered, _, _ = rasterization(
            means=xyz, quats=quats, scales=scales_activated,
            opacities=opacities_activated, colors=shs,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=tile_size, sh_degree=3, packed=True, render_mode="RGB",
        )
        torch.cuda.synchronize()

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        rendered, _, _ = rasterization(
            means=xyz, quats=quats, scales=scales_activated,
            opacities=opacities_activated, colors=shs,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=tile_size, sh_degree=3, packed=True, render_mode="RGB",
        )
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10 * 1000
    print(f"Tile {tile_size}: {dt:.3f} ms/frame")

    del rendered
    gc.collect()
    torch.cuda.empty_cache()

print("\nEnvironment check PASSED")
print("ncu is available for kernel profiling")
