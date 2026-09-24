#!/usr/bin/env python3
"""
P5.1-5.4: C1 Sort Performance — A100 baseline run (full 3DGS pipeline).
Usage: python3 p5_baseline_runner.py
Output: ~/c1_p5_baseline_results.json
"""
import os, sys, json, math
import numpy as np
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")

import gsplat
print(f"gsplat: {gsplat.__version__}")

# Verify baseline
CU_PATH = os.path.join(os.path.dirname(gsplat.__path__[0]), "gsplat", "cuda", "csrc", "IntersectTile.cu")
with open(CU_PATH) as f:
    cu = f.read()
is_baseline = 'iid << (32 + tile_n_bits)' in cu
print(f"State: BASELINE={is_baseline} C1={not is_baseline}")

env_info = {
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gsplat": gsplat.__version__,
    "state": "baseline" if is_baseline else "c1",
}

def create_scene(n_gauss, img_size, device='cuda:0'):
    """Create synthetic 3DGS scene with camera."""
    H, W = img_size
    torch.manual_seed(42)
    device = torch.device(device)
    
    # 3D Gaussians
    means = torch.randn(n_gauss, 3, device=device) * 2.0
    quats = torch.randn(n_gauss, 4, device=device)
    # Normalize quats
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)
    scales = torch.rand(n_gauss, 3, device=device) * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(n_gauss, device=device))  # in [0,1]
    colors = torch.rand(n_gauss, 3, device=device)  # RGB
    
    # Camera (looking at origin from z=distance)
    dist = 5.0
    viewmat = torch.tensor([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., dist],
        [0., 0., 0., 1.],
    ], device=device, dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    
    # Intrinsics
    fx = fy = max(W, H) * 0.8
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                     device=device, dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]
    
    return means, quats, scales, opacities, colors, viewmat, K

def time_forward(n_gauss, img_size, tile_size=16,
                 n_warmup=30, n_measured=100, n_repeats=3):
    H, W = img_size
    means, quats, scales, opacities, colors, viewmats, Ks = create_scene(n_gauss, img_size)
    
    repeat_times = []
    for rep in range(n_repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for _ in range(n_warmup):
            with torch.no_grad():
                _, _, meta = gsplat.rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities,
                    colors=colors, viewmats=viewmats, Ks=Ks,
                    width=W, height=H, tile_size=tile_size,
                    near_plane=0.01, far_plane=100.0,
                    render_mode='RGB', packed=True,
                )
        
        times = []
        for _ in range(n_measured):
            torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                _, _, meta = gsplat.rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities,
                    colors=colors, viewmats=viewmats, Ks=Ks,
                    width=W, height=H, tile_size=tile_size,
                    near_plane=0.01, far_plane=100.0,
                    render_mode='RGB', packed=True,
                )
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        
        repeat_times.append(np.array(times))
        avg = np.mean(times)
        print(f"  Rep {rep+1}: {avg:.3f}ms ± {np.std(times):.3f}ms (n={n_measured})", flush=True)
    
    all_times = np.concatenate(repeat_times)
    return {
        "mean": float(np.mean(all_times)),
        "median": float(np.median(all_times)),
        "std": float(np.std(all_times)),
        "min": float(np.min(all_times)),
        "max": float(np.max(all_times)),
        "n": int(len(all_times)),
        "n_gaussians": n_gauss,
        "img_size": list(img_size),
    }

# Test configs
test_configs = [
    (5000, (540, 960), "5K@960x540"),
    (10000, (540, 960), "10K@960x540"),
    (50000, (540, 960), "50K@960x540"),
    (100000, (540, 960), "100K@960x540"),
    (50000, (1080, 1920), "50K@1920x1080"),
]

results = {"env": env_info, "timings": {}}

for n_gauss, img_size, label in test_configs:
    print(f"\n--- {label} ({n_gauss} G @ {img_size[1]}x{img_size[0]}) ---", flush=True)
    t = time_forward(n_gauss, img_size, n_warmup=30, n_measured=100, n_repeats=3)
    results["timings"][label] = t
    print(f"  Result: {t['mean']:.3f}ms ± {t['std']:.3f}ms", flush=True)

out_path = os.path.expanduser("~/c1_p5_baseline_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")

# Summary
print("\n=== BASELINE TIMING SUMMARY ===")
print(f"{'Config':>20s} | {'Mean':>8s} | {'Std':>8s} | {'Median':>8s}")
print("-" * 55)
for _, _, lbl in test_configs:
    t = results["timings"][lbl]
    print(f"{lbl:>20s} | {t['mean']:>7.2f}ms | {t['std']:>7.2f}ms | {t['median']:>7.2f}ms")
