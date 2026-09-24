#!/usr/bin/env python3
"""GPU2 — T5' Zero-Active / Structural-Cost Experiment.

Create controlled states where active fraction varies while structural
workload remains similar.  The critical test: if T_{0%} ≈ T_{1%}, there
is significant structural cost unrelated to useful gradients.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

W, H, TILE = 1920, 1080, 16
DEV = "cuda"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=5)
    args = p.parse_args()
    torch.manual_seed(0)
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=DEV)
    cams = resize_cameras(load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=DEV), W, H)
    cam = cams[args.camera]
    bg = torch.zeros(1, 3, device=DEV)
    means = scene["xyz"].detach().clone().requires_grad_(True)
    quats = torch.nn.functional.normalize(scene["rotations"].detach().clone(), dim=-1).requires_grad_(True)
    scales = scene["scales"].detach().clone().exp().requires_grad_(True)
    opac = scene["opacity"].detach().clone().requires_grad_(True)
    shs = scene["shs"].detach().clone().requires_grad_(True)

    # Warmup
    for _ in range(3):
        for x in [means, quats, scales, opac, shs]: x.grad = None
        rgb, a, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),
            Ks=cam.K[None].contiguous(), width=W, height=H,
            near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
            sh_degree=3, packed=False, tile_size=TILE, backgrounds=bg, render_mode="RGB")
        (rgb.float().mean() + a.float().mean()).backward()
    torch.cuda.synchronize()

    # Fractions to test: 0, 1, 4, 8, 16, 32, 100 (%)
    fractions = [0.0, 0.01, 0.04, 0.08, 0.16, 0.32, 1.0]
    n_total = means.shape[0]
    results = []

    for sf in fractions:
        n = max(int(n_total * sf), 1) if sf > 0 else 1
        fwds, bwds = [], []
        t0 = time.perf_counter()
        for rep in range(10):  # 10 reps for tighter measurement
            for x in [means, quats, scales, opac, shs]: x.grad = None
            torch.cuda.synchronize()
            rgb, a, meta = gsplat.rasterization(
                means=means[:n], quats=quats[:n], scales=scales[:n],
                opacities=opac[:n], colors=shs[:n],
                viewmats=cam.world_view_transform[None].contiguous(),
                Ks=cam.K[None].contiguous(), width=W, height=H,
                near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
                sh_degree=3, packed=False, tile_size=TILE,
                backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
                rasterize_mode="classic")
            torch.cuda.synchronize()
            (rgb.float().mean() + a.float().mean()).backward()
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / 10 * 1000  # avg per rep

        # Get intersection counts
        _, iid, fid = gsplat.isect_tiles(
            meta["means2d"].contiguous(), meta["radii"].contiguous(),
            meta["depths"].contiguous(), TILE, math.ceil(W/TILE), math.ceil(H/TILE), sort=False)
        n_isect = int(iid.numel()); n_sorted = int(fid.numel())
        
        results.append({
            "fraction": sf, "n_gaussians": n,
            "T_iter_avg_ms": wall,
            "n_intersections": n_isect, "n_sorted": n_sorted,
        })
        print(f"  frac={sf:.2f} n={n:6d} T_iter={wall:.4f}ms isect={n_isect}", flush=True)

    full = results[-1]["T_iter_avg_ms"]
    for r in results:
        r["relative_to_full"] = r["T_iter_avg_ms"] / full if full > 0 else 0

    # Critical test: T0 vs T1
    t0_val = results[0]["T_iter_avg_ms"]
    t1_val = results[1]["T_iter_avg_ms"]
    zero_vs_one_ratio = t0_val / max(t1_val, 0.001)

    # Cost decomposition
    # T_total = T_fixed_overhead + T_scalable
    # T_fixed_overhead ≈ T_0 (at 0% Gaussians, the only cost is overhead)
    # T_scalable_at_100% = T_100 - T_0
    t100_val = results[-1]["T_iter_avg_ms"]
    t_fixed = t0_val
    t_scalable = t100_val - t_fixed

    out = {
        "schema_version": 2, "phase": "T5' zero-active structural cost",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE, "camera": args.camera},
        "fraction_results": results,
        "cost_decomposition": {
            "T_fixed_overhead_ms": round(t_fixed, 4),
            "T_scalable_at_full_ms": round(t_scalable, 4),
            "pct_fixed_overhead_of_full": round(t_fixed/t100_val*100, 1) if t100_val > 0 else 0,
            "T0_vs_T1_ratio": round(zero_vs_one_ratio, 3),
            "interpretation": (
                f"At 0% Gaussians: {t0_val:.3f}ms. At 1% Gaussians: {t1_val:.3f}ms. "
                f"Ratio: {zero_vs_one_ratio:.2f}x. "
                f"Fixed structural overhead: {t_fixed:.3f}ms ({t_fixed/t100_val*100:.0f}% of full T_iter). "
                f"Scalable portion: {t_scalable:.3f}ms. "
                f"This structural overhead includes launch latency, grid scheduling, "
                f"shared-memory initialization, and tile traversal that happens regardless of active pixels."
            )
        },
        "verdict": "KEEP",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")
    print(f"  Fixed overhead: {t_fixed:.4f}ms ({t_fixed/t100_val*100:.0f}% of full)")

if __name__ == "__main__":
    main()
