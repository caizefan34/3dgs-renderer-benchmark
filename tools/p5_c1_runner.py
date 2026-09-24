#!/usr/bin/env python3
"""
P5.1-5.4: C1 Sort Performance Verification — A100 C1 run.

Run this AFTER baseline runner with C1 patch applied.
Usage: python3 p5_c1_runner.py
Output: ~/c1_p5_c1_results.json
"""
import os, sys, json, math, shutil
import numpy as np
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")

import gsplat
import gsplat.cuda as gs_cuda
print(f"gsplat: {gsplat.__version__}")

# ── Verify C1 key structure ──────────────────────────────────────────
CU_PATH = os.path.join(os.path.dirname(gsplat.__path__[0]), "gsplat", "cuda", "csrc", "IntersectTile.cu")
with open(CU_PATH) as f:
    cu_content = f.read()

is_c1 = 'iid << (16 + tile_n_bits)' in cu_content
print(f"State: BASELINE={not is_c1} C1={is_c1}")
if not is_c1:
    print("ERROR: C1 patch NOT applied! Aborting.")
    sys.exit(1)

# Extract key bit info
import re
end_bit_match = re.search(r'end_bit.*?(\d+)\s*\+?\s*tile_n_bits\s*\+?\s*image_n_bits', cu_content)
tile_n_bits_match = re.search(r'tile_n_bits.*?=.*?(\d+)', cu_content)
print(f"Key info: {end_bit_match.group(0) if end_bit_match else 'not found'}")

# ── Timing helper (same as baseline) ─────────────────────────────────
def time_forward(n_gauss, img_size, tile_size=16,
                 n_warmup=30, n_measured=100, n_repeats=3):
    H, W = img_size
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    
    means2d = torch.rand(1, n_gauss, 2, device=device) * 2.0 - 1.0
    depths = torch.exp(torch.rand(1, n_gauss, device=device) * 3.0 + math.log(0.2))
    radii = torch.randint(8, 32, (1, n_gauss, 2), device=device, dtype=torch.int32)
    colors = torch.rand(1, n_gauss, 4, device=device)
    
    repeat_times = []
    for rep in range(n_repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = gs_cuda.rasterize_to_pixels_3dgs_fwd(
                    means2d=means2d, depths=depths, radii=radii,
                    colors=colors, viewport=(W, H), tile_size=tile_size,
                )
        
        times = []
        for _ in range(n_measured):
            torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                _ = gs_cuda.rasterize_to_pixels_3dgs_fwd(
                    means2d=means2d, depths=depths, radii=radii,
                    colors=colors, viewport=(W, H), tile_size=tile_size,
                )
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        
        repeat_times.append(np.array(times))
        print(f"  Rep {rep+1}: {np.mean(times):.3f}ms ± {np.std(times):.3f}ms (n={n_measured})", flush=True)
    
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

# ── Test configs (same as baseline) ───────────────────────────────────
test_configs = [
    (5000, (540, 960), "5K@960x540"),
    (10000, (540, 960), "10K@960x540"),
    (50000, (540, 960), "50K@960x540"),
    (100000, (540, 960), "100K@960x540"),
    (50000, (1080, 1920), "50K@1920x1080"),
]

results = {"env": {
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gsplat": gsplat.__version__,
    "state": "c1",
}, "timings": {}}

for n_gauss, img_size, label in test_configs:
    print(f"\n--- {label} ({n_gauss} G @ {img_size[1]}x{img_size[0]}) ---", flush=True)
    t = time_forward(n_gauss, img_size, n_warmup=30, n_measured=100, n_repeats=3)
    results["timings"][label] = t
    print(f"  Result: {t['mean']:.3f}ms ± {t['std']:.3f}ms", flush=True)

# ── Save ──────────────────────────────────────────────────────────────
out_path = os.path.expanduser("~/c1_p5_c1_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")

# ── Summary ───────────────────────────────────────────────────────────
print("\n=== C1 TIMING SUMMARY ===")
print(f"{'Config':>20s} | {'Mean':>8s} | {'Std':>8s} | {'Median':>8s}")
print("-" * 55)
for _, _, lbl in test_configs:
    t = results["timings"][lbl]
    print(f"{lbl:>20s} | {t['mean']:>7.2f}ms | {t['std']:>7.2f}ms | {t['median']:>7.2f}ms")
