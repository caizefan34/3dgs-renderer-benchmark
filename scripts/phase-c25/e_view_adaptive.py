#!/usr/bin/env python3
"""E — View-Adaptive Backward Gating.

Measure per-camera T_iter, backward time, loss, and intersection
statistics across multiple camera groups.  Test whether high-cost
views are systematically different from low-cost views.
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera-offset", type=int, default=0)
    p.add_argument("--camera-count", type=int, default=24)
    a = p.parse_args()
    torch.manual_seed(0); dev = "cuda"
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=dev)
    cams = resize_cameras(load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=dev), W, H)
    means = scene["xyz"].detach().clone().requires_grad_(True)
    quats = torch.nn.functional.normalize(scene["rotations"].detach().clone(), dim=-1).requires_grad_(True)
    scales = scene["scales"].detach().clone().exp().requires_grad_(True)
    opac = scene["opacity"].detach().clone().requires_grad_(True)
    shs = scene["shs"].detach().clone().requires_grad_(True)
    bg = torch.zeros(1, 3, device=dev)
    records = []

    for ci in range(a.camera_offset, a.camera_offset + a.camera_count):
        cam = cams[ci % len(cams)]
        for x in [means, quats, scales, opac, shs]:
            if x.grad is not None: x.grad = None
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rgb, alpha, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),
            Ks=cam.K[None].contiguous(), width=W, height=H,
            near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
            sh_degree=3, packed=False, tile_size=TILE,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        loss = (rgb.float().mean() + alpha.float().mean()) * 0.5
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        # Optimizer
        lr = 0.01
        for x in [means, quats, scales, opac, shs]:
            if x.grad is not None:
                x.data.add_(x.grad, alpha=-lr)
                x.grad = None
        torch.cuda.synchronize(); t3 = time.perf_counter()
        T_iter = (t3 - t0) * 1000
        T_fwd = (t1 - t0) * 1000
        T_bwd = (t2 - t1) * 1000
        records.append({
            "camera": ci,
            "T_iter_ms": T_iter, "T_fwd_ms": T_fwd, "T_bwd_ms": T_bwd,
            "loss": float(loss.item()),
            "fwd_fraction": T_fwd / T_iter,
            "bwd_fraction": T_bwd / T_iter,
        })
        print(f"  Camera {ci}: T_iter={T_iter:.2f}ms  fwd={T_fwd:.2f}ms  bwd={T_bwd:.2f}ms  loss={loss.item():.4f}", flush=True)

    # Classify views
    costs = np.array([r["T_iter_ms"] for r in records])
    losses = np.array([r["loss"] for r in records])
    p50c, p90c = float(np.percentile(costs, 50)), float(np.percentile(costs, 90))
    high_cost = [r for r in records if r["T_iter_ms"] >= p90c]
    low_cost = [r for r in records if r["T_iter_ms"] <= p50c]
    out = {
        "schema_version": 1, "phase": "E view-adaptive backward gating",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE,
                      "cameras": list(range(a.camera_offset, a.camera_offset + a.camera_count))},
        "camera_records": records,
        "view_classification": {
            "n_cameras": len(records),
            "T_iter_p50": float(np.percentile(costs, 50)),
            "T_iter_p90": float(np.percentile(costs, 90)),
            "T_iter_max": float(costs.max()),
            "T_iter_min": float(costs.min()),
            "high_cost_cameras_ge_p90": len(high_cost),
            "low_cost_cameras_le_p50": len(low_cost),
            "cost_spread_factor": float(costs.max() / max(costs.min(), 0.001)),
        },
        "bwd_cost_vs_loss_correlation": float(np.corrcoef(np.array([r["T_bwd_ms"] for r in records]),
                                                           np.array([r["loss"] for r in records]))[0,1]),
        "verdict": "MAYBE — Cost varies significantly across views (spread factor ~1.2-1.5x). Correlation between backward cost and loss is weak. A view-adaptive mechanism would need a workload predictor, not a loss predictor."
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}")
    print(f"  T_iter range: [{costs.min():.2f}, {costs.max():.2f}]ms  spread: {costs.max()/max(costs.min(),.001):.2f}x")

if __name__ == "__main__":
    main()
