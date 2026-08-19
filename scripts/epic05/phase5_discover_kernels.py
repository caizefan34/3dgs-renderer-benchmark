#!/usr/bin/env python3
"""
Phase 5: Discover gsplat kernel names by running a quick ncu profile.
"""
import os, sys, subprocess, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from gsplat import rasterization
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

scene_path = "data/official/mipnerf360/room/point_cloud.ply"
cam_path = "data/official/mipnerf360/room/cameras.json"
scene = load_ply(scene_path, device="cuda")
cams = load_cameras_from_json(cam_path, device="cuda")
cams = resize_cameras(cams, 1920, 1080)
cam = cams[0]

quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales_activated = torch.exp(scene["scales"]).contiguous()
opacities_activated = torch.sigmoid(scene["opacity"]).squeeze(-1).contiguous()
shs = scene["shs"].contiguous()
xyz = scene["xyz"].contiguous()

# Dump kernel names using nvprof-style: PyTorch profiler
import torch.profiler as profiler

for tile_size in [16, 32]:
    print(f"\n{'='*60}")
    print(f"Tile {tile_size} - Kernel list via PyTorch Profiler")
    print(f"{'='*60}")
    
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(3):
            rendered, _, _ = rasterization(
                means=xyz, quats=quats, scales=scales_activated,
                opacities=opacities_activated, colors=shs,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=tile_size, sh_degree=3, packed=True, render_mode="RGB",
            )
            torch.cuda.synchronize()
    
    # Print kernel table
    events = prof.key_averages(group_by_input_shape=True)
    print(f"{'Kernel':80s} {'CUDA us':>10s} {'Count':>6s} {'Avg us':>8s}")
    print("-"*110)
    for e in events:
        cu_time = e.cuda_time  # us
        cnt = e.count
        avg = cu_time / cnt if cnt else 0
        name = str(e.key)[:80]
        print(f"{name:80s} {cu_time:10.1f} {cnt:6d} {avg:8.1f}")
    
    if tile_size == 16:
        kernel_names_16 = set(str(e.key) for e in events)
    else:
        kernel_names_32 = set(str(e.key) for e in events)
        # Find which kernels are unique to each tile size
        only_16 = kernel_names_16 - kernel_names_32
        only_32 = kernel_names_32 - kernel_names_16
        if only_16:
            print(f"\nKernels only in tile16: {only_16}")
        if only_32:
            print(f"\nKernels only in tile32: {only_32}")
        
        # Top 5 kernels by total time for each tile size
        for ts, kset in [(16, kernel_names_16), (32, kernel_names_32)]:
            print(f"\nTop 5 kernels by total CUDA time (tile{ts}):")
            sorted_events = sorted(
                [e for e in events if str(e.key) in kset],
                key=lambda e: e.cuda_time, reverse=True
            )[:5]
            for e in sorted_events:
                print(f"  {str(e.key):60s} {e.cuda_time:10.1f} us")
