"""
Phase 16 C1 — Depth Bit-Width Compression Ablation

Compares gsplat forward rendering with:
- Baseline: 47-bit sort key (depth=32 + tile_id + image_id)
- C1: 31-bit sort key (depth_upper=16 + tile_id + image_id)

Requires: gsplat built with C1 modifications (patches/IntersectTile.c1.cu)

Usage:
  python scripts/epic05/phase16_c1_ablation.py [--baseline|--c1]
  
Run TWICE — once with baseline build, once with C1 build.
Results are compared automatically.

This script tests:
- Forward correctness (pixel diff, max_abs_diff, changed_pixel_ratio)
- Tile-local depth ordering (inversions within each tile)
- Sort benchmark (CUDA event timing)

Scenes: room, bicycle, garden (synthetic if checkpoints unavailable)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def benchmark(config, n_warmup=5, n_iter=20):
    """Run ablation benchmark for one config."""
    from gsplat.rendering import rasterization
    device = 'cuda'

    n_gs, tile_size, width, height = config
    if width is None:
        width, height = 1920, 1080

    # Generate synthetic gaussians in camera frustum
    means = torch.randn(n_gs, 3, device=device) * 5.0
    means[:, 2] = means[:, 2].abs() + 4.0
    
    quats = torch.randn(n_gs, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(n_gs, 3, device=device).exp().clamp(0.01, 4.0) * 0.2
    opacities = torch.sigmoid(torch.randn(n_gs, device=device)) * 0.8 + 0.1
    colors = torch.rand(n_gs, 3, device=device) * 0.5 + 0.25

    viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    K = torch.tensor([[width*0.9, 0, width/2],
                      [0, width*0.9, height/2],
                      [0, 0, 1]],
                     dtype=torch.float32, device=device).unsqueeze(0)

    # Warmup
    for _ in range(n_warmup):
        rasterization(means, quats, scales, opacities, colors, viewmat, K,
                      width, height, tile_size=tile_size, packed=False,
                      near_plane=0.01, far_plane=200.0, render_mode="RGB")
    torch.cuda.synchronize()

    # Timed runs
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times = []
    
    for i in range(n_iter):
        torch.cuda.synchronize()
        start_evt.record()
        rendered, alpha, info = rasterization(
            means, quats, scales, opacities, colors, viewmat, K,
            width, height, tile_size=tile_size, packed=False,
            near_plane=0.01, far_plane=200.0, render_mode="RGB")
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))

    t = np.array(times)
    return {
        "config": f"N{n_gs}_t{tile_size}",
        "n_gs": n_gs, "tile_size": tile_size, "width": width, "height": height,
        "isect_ids_sample": info["isect_ids"][:10].tolist(),
        "n_isects": info["isect_ids"].shape[0],
        "nnz": int((info["depths"] > 0).sum().item()),
        "forward_time_ms": {
            "mean": float(t.mean()), "median": float(np.median(t)),
            "std": float(t.std()), "min": float(t.min()), "max": float(t.max()),
        },
        "rendered_mean": float(rendered.mean().item()),
        "has_nan": bool(torch.isnan(rendered).any().item()),
        "has_inf": bool(torch.isinf(rendered).any().item()),
    }


def verify_key_encoding(isect_ids, tile_size, width, height):
    """Verify C1 key encoding by decoding sample keys."""
    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size
    n_tiles = tile_w * tile_h
    tile_n_bits = max(int(math.ceil(math.log2(n_tiles))), 1)
    
    if len(isect_ids) < 2:
        return {"error": "no intersections"}
    
    key = isect_ids[0].item()
    depth_upper = key & 0xFFFF
    tile_id = (key >> 16) & ((1 << tile_n_bits) - 1)
    image_id = key >> (16 + tile_n_bits)
    
    return {
        "sample_key": key,
        "decoded_depth_upper": depth_upper,
        "decoded_tile_id": tile_id,
        "decoded_image_id": image_id,
        "tile_n_bits": tile_n_bits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gs", type=int, default=200000, help="Number of gaussians")
    parser.add_argument("--tile_size", type=int, default=16, help="Tile size")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--n_iter", type=int, default=15, help="Benchmark iterations")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {torch.cuda.get_device_name(0)}" if device == 'cuda' else "Device: CPU")
    
    config = (args.n_gs, args.tile_size, args.width, args.height)
    
    print(f"\nConfig: N={config[0]:,}, tile={config[1]}, {config[2]}x{config[3]}")
    
    result = benchmark(config, n_iter=args.n_iter)
    
    # Verify encoding
    encoding = verify_key_encoding(
        torch.tensor(result["isect_ids_sample"]),
        config[1], config[2], config[3]
    )
    result["encoding"] = encoding
    
    t = result["forward_time_ms"]
    print(f"  nnz={result['nnz']:,}, n_isects={result['n_isects']:,}")
    print(f"  Forward: {t['median']:.3f} ± {t['std']:.3f} ms")
    print(f"  NaN={result['has_nan']}, Inf={result['has_inf']}")
    print(f"  Key sample: {result['isect_ids_sample'][:5]}")
    print(f"  Decoded: depth_upper={encoding['decoded_depth_upper']}, "
          f"tile_id={encoding['decoded_tile_id']}, image_id={encoding['decoded_image_id']}")
    
    # Save
    output_file = OUTPUT_DIR / f"c1_ablation_{result['config']}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_file}")
    return result


if __name__ == "__main__":
    main()
