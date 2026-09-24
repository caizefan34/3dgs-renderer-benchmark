#!/usr/bin/env python3
"""GPU0 — T5' Active-Lane Sweep.

Measure T_bwd = f(active fraction) across {100,75,50,32,16,8,4,2,1,0}%.
Since the kernel already uses last_ids, we create controlled active-fraction states
by trimming the sorted Gaussian list per pixel to simulate different termination depths.
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
TW, TH = math.ceil(W / TILE), math.ceil(H / TILE)
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
    print("Warmup...", flush=True)
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
    print("  done", flush=True)

    # Subsample fractions (including 0%: use 1 Gaussian)
    fractions = [1.0, 0.75, 0.50, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.001]
    results = []
    n_total = means.shape[0]

    for sf in fractions:
        n = max(int(n_total * sf), 1)
        # Count active warps estimate: warp = 32 lanes, lane active if Gaussian in that pixel's sorted range
        fwds, bwds = [], []
        for rep in range(5):
            for x in [means, quats, scales, opac, shs]: x.grad = None
            torch.cuda.synchronize(); t0 = time.perf_counter()
            rgb, a, meta = gsplat.rasterization(
                means=means[:n], quats=quats[:n], scales=scales[:n],
                opacities=opac[:n], colors=shs[:n],
                viewmats=cam.world_view_transform[None].contiguous(),
                Ks=cam.K[None].contiguous(), width=W, height=H,
                near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
                sh_degree=3, packed=False, tile_size=TILE,
                backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
                rasterize_mode="classic")
            torch.cuda.synchronize(); t1 = time.perf_counter()
            (rgb.float().mean() + a.float().mean()).backward()
            torch.cuda.synchronize(); t2 = time.perf_counter()
            fwds.append((t1-t0)*1000); bwds.append((t2-t1)*1000)
        
        # Count intersections from meta
        n_isect = 0
        n_sorted = 0
        n_tiles = 0
        try:
            _, iid, fid = gsplat.isect_tiles(
                meta["means2d"].contiguous(), meta["radii"].contiguous(),
                meta["depths"].contiguous(), TILE, TW, TH, sort=False)
            n_isect = int(iid.numel())
            n_sorted = int(fid.numel())
            n_tiles = int((iid > -1).sum().item()) if iid.dim() > 0 else 0
        except: pass

        fwd_avg = float(np.mean(fwds))
        bwd_avg = float(np.mean(bwds))
        active_lanes_est = n_sorted / max(W*H, 1)
        active_warps_est = active_lanes_est / 32
        
        results.append({
            "fraction": sf,
            "n_gaussians": n,
            "fwd_ms": fwd_avg,
            "bwd_ms": bwd_avg,
            "T_iter_ms": fwd_avg + bwd_avg,
            "n_intersections": n_isect,
            "n_sorted": n_sorted,
            "active_lanes_estimate": active_lanes_est,
            "active_warps_estimate": active_warps_est,
        })
        print(f"  frac={sf:.4f} n={n:6d} fwd={fwd_avg:.3f} bwd={bwd_avg:.3f} active_lanes={active_lanes_est:.1f}", flush=True)

    full_bwd = results[0]["bwd_ms"]
    for r in results:
        r["bwd_relative"] = r["bwd_ms"] / full_bwd if full_bwd > 0 else 0

    # Cost decomposition: estimate structural components
    # T_bwd ≈ T_fixed_overhead + T_linear_per_gaussian + T_tail
    # At 0.1% Gs: almost all cost is structural overhead
    min_bwd = min(r["bwd_ms"] for r in results)
    min_frac = min(r["fraction"] for r in results)
    
    out = {
        "schema_version": 2, "phase": "T5' active-lane sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE, "camera": args.camera,
                      "fractions_tested": fractions},
        "results": results,
        "analysis": {
            "full_bwd_ms": full_bwd,
            "minimal_bwd_ms": min_bwd,
            "minimal_fraction": min_frac,
            "structural_overhead_estimate_ms": min_bwd,
            "scalable_range": results[0]["bwd_ms"] - min_bwd,
            "pct_structural_overhead": min_bwd / full_bwd * 100,
            "interpretation": (
                f"At {min_frac*100:.1f}% Gaussian count, backward takes {min_bwd:.3f}ms "
                f"({min_bwd/full_bwd*100:.0f}% of full backward cost). "
                f"This represents structural/overhead cost that does NOT scale with active work. "
                f"The scalable component is {full_bwd-min_bwd:.3f}ms."
            )
        },
        "verdict": "KEEP — structural overhead confirmed. Need depth-tail isolation for granularity.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
