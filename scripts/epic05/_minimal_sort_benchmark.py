#!/usr/bin/env python3
"""
Minimal segmented sort benchmark.
Tests only isect_tiles(sort=True) with segmented=False vs segmented=True
on a smaller workload (room, 1080p) using already-validated camera loading.
"""

import sys, os, math, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import torch
import torch.nn.functional as F
from benchmark_framework.scene import load_ply
from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode
from gsplat import rasterization

DEVICE = torch.device("cuda:0")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimal test: just time isect_tiles directly
print("Loading scene...")
scene = load_ply(os.path.join(REPO_ROOT, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)

# Use same camera format as validated phase12
from benchmark_framework import load_cameras_from_json, resize_cameras
cam_path = os.path.join(REPO_ROOT, "data", "official", "mipnerf360", "room", "cameras.json")
cameras = load_cameras_from_json(cam_path, device="cpu")
W, H = 1920, 1080
cameras = resize_cameras(cameras, W, H)
cam = cameras[0]

# Prepare params with correct activations
N = scene['xyz'].shape[0]
params = {
    "xyz": scene['xyz'].detach().clone().to(DEVICE),
    "rotations": F.normalize(scene['rotations'], dim=-1).contiguous().to(DEVICE),
    "scales": torch.exp(scene['scales']).contiguous().to(DEVICE),  # exp for log-scale
    "opacity": torch.sigmoid(scene['opacity']).contiguous().to(DEVICE),
    "shs": scene['shs'].contiguous().to(DEVICE),
    "sh_degree": scene['sh_degree'],
}
viewmat = cam.world_view_transform.to(DEVICE).unsqueeze(0)
K = cam.K.to(DEVICE).unsqueeze(0)
print(f"N={N:,}, viewmat={viewmat.shape}, K={K.shape}")

# Run one forward pass to get isect_tiles timing
for seg_name, seg_val in [('baseline', False), ('segmented', True)]:
    print(f"\n--- {seg_name} ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        rendered, alpha, meta = rasterization(
            means=params["xyz"].unsqueeze(0),
            quats=params["rotations"].unsqueeze(0),
            scales=params["scales"].unsqueeze(0),
            opacities=params["opacity"].unsqueeze(0),
            colors=params["shs"].unsqueeze(0),
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W, height=H,
            tile_size=16, packed=True, sh_degree=params["sh_degree"],
            segmented=seg_val,
        )
    
    n_isects = meta['isect_ids'].shape[0]
    peak = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"  n_isects={n_isects:,}")
    print(f"  peak_mem={peak:.0f} MB")
    print(f"  rendered={rendered.shape}")
    
    if seg_name == 'baseline':
        ref = rendered.clone()
    else:
        diff = torch.max(torch.abs(rendered - ref)).item()
        print(f"  max_pixel_diff={diff:.10f}")

# Now time the sort specifically
print("\n\n=== Timing sort ===")
for tile_size in [16, 20, 32]:
    for seg_name, seg_val in [('baseline', False), ('segmented', True)]:
        times = []
        with torch.no_grad():
            for _ in range(5):
                rendered, alpha, meta = rasterization(
                    means=params["xyz"].unsqueeze(0),
                    quats=params["rotations"].unsqueeze(0),
                    scales=params["scales"].unsqueeze(0),
                    opacities=params["opacity"].unsqueeze(0),
                    colors=params["shs"].unsqueeze(0),
                    viewmats=viewmat.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=W, height=H,
                    tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
                    segmented=seg_val,
                )
        
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rendered, alpha, meta = rasterization(
                means=params["xyz"].unsqueeze(0),
                quats=params["rotations"].unsqueeze(0),
                scales=params["scales"].unsqueeze(0),
                opacities=params["opacity"].unsqueeze(0),
                colors=params["shs"].unsqueeze(0),
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W, height=H,
                tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
                segmented=seg_val,
            )
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        
        t = torch.tensor(times)
        print(f"  tile{tile_size} {seg_name}: avg={t.mean():.2f}ms med={t.median():.2f}ms std={t.std():.2f}ms n_isects={meta['isect_ids'].shape[0]:,}")
