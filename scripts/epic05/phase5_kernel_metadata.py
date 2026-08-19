#!/usr/bin/env python3
"""
Phase 5: Extract kernel resource metadata via PyTorch CUPTI interface.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.profiler as profiler
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

for tile_size in [16, 32]:
    print(f"\n{'='*70}")
    print(f"  Tile {tile_size} — Kernel Resource Metadata")
    print(f"{'='*70}")
    
    # Use detailed profiler with stack information
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(3):
            rendered, _, _ = rasterization(
                means=xyz, quats=quats, scales=scales,
                opacities=ops, colors=shs,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=tile_size, sh_degree=3, packed=True, render_mode="RGB",
            )
            torch.cuda.synchronize()
    
    # Extract per-kernel details
    gsplat_kernels = {}
    for e in prof.events():
        name = str(e.key)
        if "gsplat" in name and "rasterize_to_pixels" in name:
            kdata = {
                "device_time_us": e.device_time,
                "count": e.count,
                "avg_us": e.device_time / e.count if e.count else 0,
            }
            # Try to get input shapes
            try:
                kdata["input_shapes"] = str(e.input_shapes)
            except:
                pass
            # Try to get call stack
            try:
                kdata["stack"] = str(e.stack[:3])
            except:
                pass
            gsplat_kernels[name[:80]] = kdata
    
    for kname, kdata in gsplat_kernels.items():
        print(f"\n  Kernel: {kname}")
        for key, val in kdata.items():
            print(f"    {key}: {val}")
    
    # Also scan all CUDA events for gsplat
    print(f"\n  All gsplat kernels:")
    for e in prof.events():
        name = str(e.key)
        if "gsplat" in name:
            shapes = getattr(e, 'input_shapes', None)
            print(f"    {name[:70]:70s} time={e.device_time:10.1f}us count={e.count:3d} shapes={shapes}")
    
    print()
