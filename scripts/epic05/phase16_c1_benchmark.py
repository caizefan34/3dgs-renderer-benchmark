"""
Phase 16 C1 — High-Traffic Sort Benchmark v4

Corrected camera convention (OpenGL: camera looks down +z).
Generates 2M+ n_isects for meaningful timing.
"""

import json, os, math, time
from pathlib import Path
import numpy as np
import torch

torch.manual_seed(42)
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def benchmark(n_gaussians=200000, width=1920, height=1080, tile_size=16, n_warmup=5, n_iter=20):
    """Benchmark forward rasterization."""
    from gsplat.rendering import rasterization
    device = 'cuda'

    # Gaussians in front of camera (camera at origin, looks down +z)
    means = torch.randn(n_gaussians, 3, device=device) * 5.0
    means[:, 2] = means[:, 2].abs() + 4.0  # z in [4, ∞) — in front of camera
    
    quats = torch.randn(n_gaussians, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(n_gaussians, 3, device=device).exp().clamp(0.005, 2.0) * 0.15
    opacities = torch.sigmoid(torch.randn(n_gaussians, device=device)) * 0.8 + 0.1
    colors = torch.rand(n_gaussians, 3, device=device) * 0.5 + 0.25

    viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    K = torch.tensor([[width*0.9, 0, width/2], [0, width*0.9, height/2], [0, 0, 1]],
                     dtype=torch.float32, device=device).unsqueeze(0)

    # Warmup
    for _ in range(n_warmup):
        rasterization(means, quats, scales, opacities, colors, viewmat, K,
                      width, height, tile_size=tile_size, packed=False,
                      near_plane=0.01, far_plane=200.0, render_mode="RGB")
    torch.cuda.synchronize()

    # First run to get info
    _, _, info = rasterization(means, quats, scales, opacities, colors, viewmat, K,
                                width, height, tile_size=tile_size, packed=False,
                                near_plane=0.01, far_plane=200.0, render_mode="RGB")
    isect_ids = info["isect_ids"]
    depths = info["depths"]
    n_isects = isect_ids.shape[0]
    nnz = int((depths > 0).sum().item())
    print(f"  nnz={nnz:,} / {n_gaussians:,}, n_isects={n_isects:,}")

    # Timed runs
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times = []
    
    for i in range(n_iter):
        torch.cuda.synchronize()
        start_evt.record()
        rasterization(means, quats, scales, opacities, colors, viewmat, K,
                      width, height, tile_size=tile_size, packed=False,
                      near_plane=0.01, far_plane=200.0, render_mode="RGB")
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))

    t = np.array(times)
    tw, th = info["tile_width"], info["tile_height"]
    tile_n_bits = max(int(math.ceil(math.log2(tw * th))), 1)
    
    return {
        "n_gaussians": n_gaussians, "nnz": nnz, "n_isects": n_isects,
        "tile_size": tile_size, "width": width, "height": height,
        "tile_width": tw, "tile_height": th, "tile_n_bits": tile_n_bits,
        "theoretical_radix_passes": (16 + tile_n_bits + 1 + 3) // 4,
        "forward_time_ms": {
            "mean": float(t.mean()), "median": float(np.median(t)),
            "std": float(t.std()), "min": float(t.min()), "max": float(t.max()),
        },
        "key_sample": isect_ids[:5].tolist(),
    }


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    configs = [(100000, 16), (200000, 16), (500000, 16)]
    results = {"device": torch.cuda.get_device_name(0), "scenarios": []}
    
    for n_gs, tile_sz in configs:
        print(f"\n--- N={n_gs:,}, tile={tile_sz} ---")
        r = benchmark(n_gs, tile_size=tile_sz)
        r["label"] = f"N{n_gs}_t{tile_sz}"
        results["scenarios"].append(r)
        t = r["forward_time_ms"]
        print(f"  fwd={t['median']:.3f}±{t['std']:.3f}ms  n_isects={r['n_isects']:,}")
        print(f"  passes={r['theoretical_radix_passes']} key_sample={r['key_sample']}")

    output_file = OUTPUT_DIR / "c1_benchmark_result.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
