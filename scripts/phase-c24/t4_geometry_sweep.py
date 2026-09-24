#!/usr/bin/env python3
"""T4 — Backward-Specific Execution Geometry.

Measure backward iteration time at different tile sizes.  This test
runs a full training step (forward + backward + update) at each tile
size and reports the wall-clock iteration time.

Primary metric: end-to-end training iteration time T_iter.
Secondary: T_F+B, backward kernel time.
"""
from __future__ import annotations
import argparse, json, math, time, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

W, H = 1920, 1080
TILE_VALUES = [8, 12, 16, 20, 24, 28]  # tile 32 excluded: backward kernel exceeds register/resource limit

def time_training_step(means, quats, scales, opac, shs, cam, tile_size, bg):
    """Time one forward+backward+update iteration, wall-clock."""
    for x in (means, quats, scales, opac, shs):
        if x.grad is not None:
            x.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rgb, alpha, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H, near_plane=.01, far_plane=1e10,
        radius_clip=0., eps2d=.3, sh_degree=3, packed=False, tile_size=tile_size,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    (rgb.float().mean() + alpha.float().mean()).backward()
    # Simple SGD-style update to make this a complete training iteration
    lr = 0.01
    for param in [means, quats, scales, opac, shs]:
        if param.grad is not None:
            param.data.add_(param.grad, alpha=-lr)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera-offset", type=int, default=0)
    p.add_argument("--camera-count", type=int, default=4)
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=dev)
    cameras = resize_cameras(
        load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=dev),
        W, H)
    means = scene["xyz"].detach().clone().requires_grad_(True)
    quats = torch.nn.functional.normalize(scene["rotations"].detach().clone(), dim=-1).requires_grad_(True)
    scales = scene["scales"].detach().clone().exp().requires_grad_(True)
    opac = scene["opacity"].detach().clone().requires_grad_(True)
    shs = scene["shs"].detach().clone().requires_grad_(True)
    bg = torch.zeros(1, 3, device=dev)

    cam_records = []
    all_itimes = {ts: [] for ts in TILE_VALUES}

    for ci in range(a.camera_offset, a.camera_offset + a.camera_count):
        cam = cameras[ci % len(cameras)]
        for tile_size in TILE_VALUES:
            try:
                # Warmup
                for _ in range(2):
                    time_training_step(means, quats, scales, opac, shs, cam, tile_size, bg)
                # Measured run
                times = []
                for _ in range(5):
                    t = time_training_step(means, quats, scales, opac, shs, cam, tile_size, bg)
                    times.append(t)
                t_mean = float(np.mean(times))
                all_itimes[tile_size].append(t_mean)
                print(f"  Camera {ci}, tile={tile_size}: T_iter={t_mean:.1f}ms", flush=True)
            except Exception as ex:
                print(f"  Camera {ci}, tile={tile_size}: SKIPPED ({ex})", flush=True)

    aggregate = {}
    for ts, vals in all_itimes.items():
        arr = np.array(vals)
        aggregate[str(ts)] = {
            "mean_T_iter_ms": float(arr.mean()),
            "p50_T_iter_ms": float(np.percentile(arr, 50)),
            "min_T_iter_ms": float(arr.min()),
            "max_T_iter_ms": float(arr.max()),
        }
    best_ts = min(TILE_VALUES, key=lambda ts: aggregate[str(ts)]["mean_T_iter_ms"])
    baseline_ts = 16
    speedup = aggregate[str(baseline_ts)]["mean_T_iter_ms"] / aggregate[str(best_ts)]["mean_T_iter_ms"] - 1

    out = {
        "schema_version": 1,
        "phase": "T4 backward execution geometry sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}",
                      "tile_values": TILE_VALUES,
                      "cameras": list(range(a.camera_offset, a.camera_offset + a.camera_count))},
        "per_fixed_checkpoint": True,
        "measurement": "full training iteration (forward+backward+update)",
        "iterations_per_config": 5,
        "warmup": 2,
        "baseline_tile_size": baseline_ts,
        "best_tile_size": best_ts,
        "tile_summary": aggregate,
        "baseline_T_iter_ms": aggregate[str(baseline_ts)]["mean_T_iter_ms"],
        "best_T_iter_ms": aggregate[str(best_ts)]["mean_T_iter_ms"],
        "speedup_vs_baseline_pct": speedup * 100,
        "verdict": (
            "DROP" if speedup < 0.03
            else "MAYBE" if speedup < 0.10
            else "KEEP_CANDIDATE"
        ),
    }
    if speedup >= 0.03:
        out["interpretation"] = (
            f"Tile {best_ts} achieves {speedup*100:.1f}% training iteration speedup over tile 16. "
            "This is a pure execution geometry change requiring no algorithm modification. "
            "However, the speedup is achieved by changing only the tile size — both forward and backward use it. "
            "True asymmetric execution (different fwd/bwd tile) is not testable without kernel changes."
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}")
    print(f"Best tile: {best_ts}, baseline (16): {aggregate[str(baseline_ts)]['mean_T_iter_ms']:.1f}ms, best: {aggregate[str(best_ts)]['mean_T_iter_ms']:.1f}ms, speedup: {speedup*100:.1f}%")

if __name__ == "__main__":
    main()
