#!/usr/bin/env python3
"""
Phase 7C — Synthetic microbenchmarks for tile_size mechanism hypotheses.

All Gaussians always processed (packed=False / dense mode → no frustum culling).
Batched CUDA event timing with BATCH=30 iterations per event pair.

Benchmarks:
  M1a: Forward-only, varying N (1K → 400K)
  M6:  Sub-pixel Gaussians (radii << 1 px)

Output: results/epic05/phase7c/microbench_results.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase7c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
BATCH = 30
N_REPEAT = 5  # averages over 5 × BATCH = 150 calls per point


def make_camera(W=1920, H=1080):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))  # 50 deg HFOV
    viewmat = torch.eye(4, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    t = torch.eye(4, device=DEVICE, dtype=torch.float32)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor([[fx, 0.0, W / 2.0],
                      [0.0, fy, H / 2.0],
                      [0.0, 0.0, 1.0]], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    return viewmat, K, W, H


def make_cloud(N, scale_mean=0.3, scale_std=0.2):
    """Gaussians filling the view frustum with large enough scales."""
    xyz = (torch.rand(N, 3, device=DEVICE) - 0.5) * 4.0  # ±2 units
    q = torch.randn(N, 4, device=DEVICE)
    q = q / q.norm(dim=-1, keepdim=True)
    s = (torch.randn(N, 3, device=DEVICE) * scale_std + scale_mean).abs().clamp(min=0.01)
    o = torch.sigmoid(torch.randn(N, device=DEVICE) * 0.5 + 2.0)
    shs = torch.randn(N, 16, 3, device=DEVICE) * 0.01
    return xyz, q, s, o, shs


def _forward_batched(gens, vm, K, W, H, ts, packed=False):
    """BATCH forward passes, return ms/iter. Default dense mode."""
    from gsplat import rasterization
    xyz, q, s, o, c = gens
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(BATCH):
        rasterization(xyz, q, s, o, c, vm, K, W, H,
                      tile_size=ts, packed=packed, sh_degree=3,
                      radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    ev_e.record()
    ev_e.synchronize()
    return ev_s.elapsed_time(ev_e) / BATCH


def _fwd_bwd_batched(gens, vm, K, W, H, ts, packed=False):
    """BATCH/2 forward+backward passes, return ms/iter."""
    from gsplat import rasterization
    xyz, q, s, o, c = gens
    xyz_p = xyz.clone().requires_grad_(True)
    n = max(1, BATCH // 2)
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(n):
        r, _, _ = rasterization(xyz_p, q, s, o, c, vm, K, W, H,
                                tile_size=ts, packed=packed, sh_degree=3,
                                radius_clip=0.0, eps2d=0.1, render_mode="RGB")
        r.sum().backward()
    ev_e.record()
    ev_e.synchronize()
    return ev_s.elapsed_time(ev_e) / n


def run_m1a(results: dict):
    """Forward-only in dense AND packed mode, varying N."""
    print("\n  M1a: Forward vs N (tile16 / tile32)")
    vm, K, W, H = make_camera()
    Ns = [1000, 10000, 50000, 200000, 400000]
    data = {}
    for N in Ns:
        print(f"    N={N:>6,}", end="")
        gens = make_cloud(N)
        for packed in (False, True):
            label = "packed" if packed else "dense"
            for ts in (16, 32):
                v = []
                for _ in range(N_REPEAT):
                    v.append(_forward_batched(gens, vm, K, W, H, ts, packed=packed))
                data.setdefault(N, {}).setdefault(label, {})[f"t{ts}"] = {
                    "ms": float(np.mean(v)), "std": float(np.std(v))
                }
                print(f"  {label}t{ts}:{np.mean(v):.2f}±{np.std(v):.2f}", end="")
            r = data[N][label]["t16"]["ms"] / data[N][label]["t32"]["ms"]
            data[N][label]["speedup"] = r
            print(f"  ratio={r:.3f}x", end="  ")
        print()
    results["M1a"] = {"config": {"N": Ns, "BATCH": BATCH}, "data": {str(k): v for k, v in data.items()}}


def run_m7_fwd_bwd(results: dict):
    """Forward+backward timing (dense + packed), varying N."""
    print("\n  M7: Forward+backward vs N (tile16 / tile32)")
    vm, K, W, H = make_camera()
    Ns = [50000, 200000, 400000]
    data = {}
    for N in Ns:
        print(f"    N={N:>6,}", end="")
        gens = make_cloud(N)
        for packed in (False, True):
            label = "packed" if packed else "dense"
            for ts in (16, 32):
                v = []
                for _ in range(N_REPEAT):
                    v.append(_fwd_bwd_batched(gens, vm, K, W, H, ts, packed=packed))
                data.setdefault(N, {}).setdefault(label, {})[f"t{ts}"] = {
                    "ms": float(np.mean(v)), "std": float(np.std(v))
                }
                print(f"  {label}t{ts}:{np.mean(v):.2f}±{np.std(v):.2f}", end="")
            r = data[N][label]["t16"]["ms"] / data[N][label]["t32"]["ms"]
            data[N][label]["speedup"] = r
            print(f"  ratio={r:.3f}x", end="  ")
        print()
    results["M7"] = {"config": {"N": Ns, "BATCH": max(1, BATCH // 2)}, "data": {str(k): v for k, v in data.items()}}


def run_m6(results: dict):
    """Sub-pixel Gaussians (tiny scales) -> no tile overlap."""
    print("\n  M6: Sub-pixel Gaussians (radii << 1 px)")
    vm, K, W, H = make_camera()
    N = 200000
    # Tiny scales: all Gaussians project to sub-pixel
    gens = make_cloud(N, scale_mean=0.001, scale_std=0.0005)
    data = {}
    for ts in (16, 32):
        v = []
        for _ in range(N_REPEAT):
            v.append(_forward_batched(gens, vm, K, W, H, ts))
        data[f"t{ts}"] = {"ms": float(np.mean(v)), "std": float(np.std(v))}
        print(f"    t{ts}: {np.mean(v):.4f}±{np.std(v):.4f}ms")
    data["speedup"] = data["t16"]["ms"] / data["t32"]["ms"]
    print(f"    ratio={data['speedup']:.3f}x")
    results["M6"] = {"config": {"N": N, "mode": "dense", "desc": "subpixel"}, "data": data}


def main():
    print("=" * 60)
    print("  Phase 7C — Synthetic Microbenchmarks")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  BATCH={BATCH}, N_REPEAT={N_REPEAT}")
    print(f"  Mode: dense (packed=False) — no frustum culling")
    print("=" * 60)

    r = {"gpu": torch.cuda.get_device_name(0), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    run_m1a(r)
    run_m7_fwd_bwd(r)
    run_m6(r)

    out = OUTPUT_DIR / "microbench_results.json"
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
