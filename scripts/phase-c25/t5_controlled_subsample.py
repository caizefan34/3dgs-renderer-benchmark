#!/usr/bin/env python3
"""T5' — Controlled subsampling experiment.

Directly tests the sparse-tail hypothesis by measuring backward kernel
time across artificially subsampled Gaussian sets that produce different
active-lane fractions.

Key question: T_backward = f(active_lane_fraction)
If cost stays high as active fraction drops toward 1-5%, sparse-tail
reorganization is justified. If cost collapses proportionally, current
kernel handles sparsity efficiently.
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

def measure_iter(cam, means, quats, scales, opac, shs, bg):
    for x in [means, quats, scales, opac, shs]:
        if x.grad is not None: x.grad = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    rgb, alpha, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H,
        near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
        sh_degree=3, packed=False, tile_size=TILE,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    torch.cuda.synchronize(); t1 = time.perf_counter()
    (rgb.float().mean() + alpha.float().mean()).backward()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=5)
    args = p.parse_args()
    torch.manual_seed(0); dev = "cuda"
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=dev)
    cams = resize_cameras(load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=dev), W, H)
    cam = cams[args.camera]
    bg = torch.zeros(1, 3, device=dev)

    # Full set
    means = scene["xyz"].detach().clone().requires_grad_(True)
    quats = torch.nn.functional.normalize(scene["rotations"].detach().clone(), dim=-1).requires_grad_(True)
    scales = scene["scales"].detach().clone().exp().requires_grad_(True)
    opac = scene["opacity"].detach().clone().requires_grad_(True)
    shs = scene["shs"].detach().clone().requires_grad_(True)

    print("Warming up GPU...")
    fwd, bwd, meta = measure_iter(cam, means, quats, scales, opac, shs, bg)
    print(f"  Warmup: fwd={fwd:.2f}ms bwd={bwd:.2f}ms")

    # Subsample fractions - from dense to very sparse
    subsample_fracs = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01]
    results = []

    for sf in subsample_fracs:
        n = max(int(means.shape[0] * sf), 100)
        fwd_ms = []; bwd_ms = []
        for rep in range(5):
            fm, bm, m = measure_iter(cam, means[:n], quats[:n], scales[:n],
                                       opac[:n], shs[:n], bg)
            fwd_ms.append(fm); bwd_ms.append(bm)
        fwd_avg = float(np.mean(fwd_ms))
        bwd_avg = float(np.mean(bwd_ms))
        results.append({
            "subsample_fraction": sf,
            "n_gaussians": n,
            "fwd_ms_mean": fwd_avg,
            "bwd_ms_mean": bwd_avg,
            "Titer_ms_mean": fwd_avg + bwd_avg,
            "bwd_fraction_of_baseline": bwd_avg / bwd_ms[0] if sf == 1.0 and bwd_ms else None
        })
        print(f"  sf={sf:.2f} n={n:7d}  fwd={fwd_avg:.3f}ms  bwd={bwd_avg:.3f}ms", flush=True)

    # Normalize everything relative to full (100%) run
    full_bwd = results[0]["bwd_ms_mean"]
    for r in results:
        r["bwd_relative"] = r["bwd_ms_mean"] / full_bwd if full_bwd > 0 else 0

    out = {
        "schema_version": 1,
        "phase": "T5' controlled subsampling experiment",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE, "camera": args.camera,
                      "subsample_fractions": subsample_fracs},
        "results": results,
        "interpretation": (
            "If bwd_relative tracks subsample_fraction linearly, backward cost scales "
            "proportionally with Gaussian count and active lanes — kernel already handles "
            "sparsity efficiently, no sparse-tail mechanism needed. "
            "If bwd_relative stays high as subsample_fraction drops (e.g., at 1% Gaussians "
            "backward is still >10% of full cost), the sparse tail imposes disproportionate "
            "cost and reorganization is justified."
        ),
        "verdict": "PENDING"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")
    # Print key comparison
    for r in results:
        if r["subsample_fraction"] <= 0.1:
            print(f"  At {r['subsample_fraction']*100:.0f}% Gaussians: backward = {r['bwd_relative']*100:.1f}% of full cost")

if __name__ == "__main__":
    import numpy as np
    main()
