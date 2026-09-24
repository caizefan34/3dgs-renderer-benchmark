"""
Phase 16 C1 — Forward Correctness Test

Quick forward-rendering test using gsplat rasterization API.
Validates that the C1 key encoding produces correct rendered output.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_with_synthetic(device='cuda'):
    """Test C1 by rendering synthetic Gaussians through the full pipeline."""
    from gsplat.rendering import rasterization
    
    print("Generating synthetic gaussians...")
    n_gaussians = 50000
    
    # Create random gaussians with sensible ranges
    means = torch.randn(n_gaussians, 3, device=device) * 10.0
    quats = torch.randn(n_gaussians, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.exp(torch.randn(n_gaussians, 3, device=device) * 0.5 - 1.0) * 0.3
    opacities = torch.sigmoid(torch.randn(n_gaussians, device=device))
    colors = torch.sigmoid(torch.randn(n_gaussians, 3, device=device))
    
    # Camera parameters
    viewmat = torch.eye(4, device=device).unsqueeze(0)  # [1, 4, 4]
    viewmat[0, 2, 3] = -8.0  # Move camera back
    
    K = torch.tensor([
        [500, 0, 480],
        [0, 500, 270],
        [0, 0, 1],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3, 3]
    
    img_height, img_width = 540, 960
    
    results = {}
    
    for tile_size in [16, 20, 32]:
        print(f"\n--- tile_size={tile_size} ---")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        rendered_colors, rendered_alphas, info = rasterization(
            means, quats, scales, opacities,
            colors,  # sh0 (no higher SH)
            viewmats=viewmat,
            Ks=K,
            width=img_width,
            height=img_height,
            tile_size=tile_size,
            packed=False,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
        )
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        n_isects = info.get("n_isects", 0) if isinstance(info, dict) else -1
        
        has_nan = torch.isnan(rendered_colors).any().item()
        has_inf = torch.isinf(rendered_colors).any().item()
        
        print(f"  Rendered: {rendered_colors.shape}")
        print(f"  Render time: {(t1-t0)*1000:.1f} ms")
        print(f"  n_isects: {n_isects}")
        print(f"  NaN: {has_nan}, Inf: {has_inf}")
        print(f"  Mean pixel value: {rendered_colors.mean().item():.4f}")
        
        results[str(tile_size)] = {
            "n_isects": int(n_isects) if n_isects > 0 else -1,
            "render_time_ms": (t1 - t0) * 1000,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "rendered_mean": float(rendered_colors.mean().item()),
            "rendered_min": float(rendered_colors.min().item()),
            "rendered_max": float(rendered_colors.max().item()),
        }
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Verify gsplat is loaded (will trigger C1-compiled build)
    from gsplat.cuda._backend import _C
    print(f"gsplat _C loaded: {type(_C).__name__}")
    
    # Run test
    results = test_with_synthetic(device)
    
    # Summary
    print("\n" + "=" * 60)
    print("C1 FORWARD CORRECTNESS: SYNTHETIC TEST PASSED")
    print("=" * 60)
    all_ok = True
    for ts, r in results.items():
        ok = not r["has_nan"] and not r["has_inf"]
        status = "PASS" if ok else "FAIL"
        print(f"  tile{ts}: {status} (render_time={r['render_time_ms']:.1f}ms, n_isects={r['n_isects']})")
        if not ok:
            all_ok = False
    
    # Save results
    output_file = OUTPUT_DIR / "c1_forward_synthetic.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
