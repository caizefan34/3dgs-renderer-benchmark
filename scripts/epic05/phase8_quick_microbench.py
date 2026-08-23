#!/usr/bin/env python3
"""
Phase 8 — Quick real-scene microbenchmark: forward pass only.
Measures tile16 vs tile32 on a single frozen checkpoint (room_t16_iter5000).
"""
from __future__ import annotations
import json, math, sys, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from gsplat import rasterization

DEVICE = 'cuda'
BATCH, N_REPEAT, WARMUP = 10, 3, 2

cp = torch.load(
    'results/epic05/phase7/phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt',
    map_location=DEVICE, weights_only=False)
ms = cp['model_state']
xyz = ms['xyz']
rot = ms['rotations']
scales = ms['scales']
opac = ms['opacity'].squeeze(1)
shs = ms['shs']
N = xyz.shape[0]

W, H = 1920, 1080
fx = W / (2.0 * math.tan(math.radians(25)))
viewmat = torch.eye(4, device=DEVICE, dtype=torch.float32).unsqueeze(0)
viewmat[0, 2, 3] = -5.0
K = torch.tensor([[fx, 0, W/2.0], [0, fx, H/2.0], [0, 0, 1]],
                 device=DEVICE, dtype=torch.float32).unsqueeze(0)

print(f"Loaded {N} Gaussians from room_t16_iter5000")
print(f"SH degree: {ms['sh_degree']}")

results = []

for ts in [16, 32]:
    print(f"\n--- tile_size={ts} ---")
    for w in range(WARMUP):
        rendered, alpha, meta = rasterization(
            means=xyz, quats=rot, scales=scales, opacities=opac, colors=shs,
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=ts, packed=True, sh_degree=ms['sh_degree'])
        torch.cuda.synchronize()
    print(f"  Warmup done", flush=True)

    all_times = []
    for r in range(N_REPEAT):
        for b in range(BATCH):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            rendered, alpha, meta = rasterization(
                means=xyz, quats=rot, scales=scales, opacities=opac, colors=shs,
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=ts, packed=True, sh_degree=ms['sh_degree'])
            end_ev.record()
            torch.cuda.synchronize()
            all_times.append(start_ev.elapsed_time(end_ev))
        print(f"  Repeat {r+1}/{N_REPEAT} done", flush=True)

    arr = np.array(all_times)
    tpg = meta['tiles_per_gauss']

    entry = {
        "tile_size": ts,
        "num_gaussians": N,
        "forward_ms_mean": float(arr.mean()),
        "forward_ms_std": float(arr.std()),
        "forward_ms_cv": float(arr.std() / arr.mean()),
        "tiles_per_gauss_mean": float(tpg.float().mean().item()),
        "tiles_per_gauss_std": float(tpg.float().std().item()),
        "tiles_per_gauss_median": float(tpg.float().median().item()),
        "tiles_per_gauss_max": int(tpg.max().item()),
        "total_intersections": int(tpg.sum().item()),
        "tile_grid": f"{meta['tile_width']}x{meta['tile_height']}",
    }
    results.append(entry)
    print(f"  Forward mean: {entry['forward_ms_mean']:.2f} ± {entry['forward_ms_std']:.2f} ms")
    print(f"  Tiles/Gaussian: {entry['tiles_per_gauss_mean']:.2f} ± {entry['tiles_per_gauss_std']:.2f}")
    print(f"  Total intersections: {entry['total_intersections']}")

# Compute ratio
t16 = results[0]
t32 = results[1]
ratio = t16['forward_ms_mean'] / t32['forward_ms_mean']
print(f"\n>>> Ratio tile16/tile32: {ratio:.4f}")
print(f">>> tile32 speedup: {1/ratio:.4f}x")

print(f"\n{'='*60}")
print(f"  CONCLUSION")
print(f"{'='*60}")
if ratio > 1.05:
    print(f"  tile32 is {1/ratio:.2f}x faster on REAL checkpoint (900K Gs)")
elif ratio < 0.95:
    print(f"  tile16 is {ratio:.2f}x faster on REAL checkpoint (900K Gs)")
else:
    print(f"  tile16 ≈ tile32 (ratio={ratio:.4f}) — within noise")
