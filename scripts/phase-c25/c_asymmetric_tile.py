#!/usr/bin/env python3
"""C — Forward/Backward Asymmetric Tile Policy.

Tests different tile sizes for full training iteration across cameras.
Also analyzes whether per-camera optimal tile size is predictable from
workload statistics.
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
TILE_VALS = [8, 12, 16, 20, 24, 28]

def measure(cam, tile, means, quats, scales, opac, shs, bg):
    for x in [means, quats, scales, opac, shs]:
        if x.grad is not None: x.grad = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    rgb, a, _ = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H,
        near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
        sh_degree=3, packed=False, tile_size=tile,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    (rgb.float().mean() + a.float().mean()).backward()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    return (t1 - t0) * 1000  # T_iter ms

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--camera-offset", type=int, default=0)
    p.add_argument("--camera-count", type=int, default=6)
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
    records = []; tiles_opt = {ts: [] for ts in TILE_VALS}

    for ci in range(a.camera_offset, a.camera_offset + a.camera_count):
        cam = cams[ci % len(cams)]
        # Compute workload stats for this camera (tile=16)
        torch.set_grad_enabled(False)
        _, _, meta = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),
            Ks=cam.K[None].contiguous(), width=W, height=H,
            near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
            sh_degree=3, packed=False, tile_size=16,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        tw16 = math.ceil(W / 16); th16 = math.ceil(H / 16)
        _, iid, _ = gsplat.isect_tiles(
            meta["means2d"].contiguous(), meta["radii"].contiguous(),
            meta["depths"].contiguous(), 16, tw16, th16, sort=False)
        ioff = gsplat.isect_offset_encode(iid, 1, tw16, th16)
        ioff_flat = ioff[0].reshape(-1).cpu().numpy()
        tile_lengths = np.diff(np.concatenate([ioff_flat, [int(iid.numel())]]))
        nz = tile_lengths[tile_lengths > 0]
        torch.set_grad_enabled(True)

        cam_row = {
            "camera": ci,
            "intersection_stats": {
                "mean": float(nz.mean()), "p50": float(np.percentile(nz, 50)),
                "p90": float(np.percentile(nz, 90)), "p95": float(np.percentile(nz, 95)),
                "p99": float(np.percentile(nz, 99)), "max": float(nz.max() if len(nz) else 0),
                "nz_tiles": int(len(nz))
            }
        }
        for ts in TILE_VALS:
            ts_ms = []
            for _ in range(5):
                ts_ms.append(measure(cam, ts, means, quats, scales, opac, shs, bg))
            t_mean = float(np.mean(ts_ms))
            cam_row[f"tile_{ts}_Titer_ms"] = t_mean
            tiles_opt[ts].append(t_mean)
            print(f"  cam {ci} tile {ts}: {t_mean:.2f}ms", flush=True)
        records.append(cam_row)

    # Find best tile for each camera
    for r in records:
        best_ts = min(TILE_VALS, key=lambda ts: r[f"tile_{ts}_Titer_ms"])
        r["best_tile"] = best_ts

    agg = {}
    for ts, vals in tiles_opt.items():
        arr = np.array(vals)
        agg[str(ts)] = {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50))}

    baseline_16 = agg["16"]["mean"]
    best_ts = min(TILE_VALS, key=lambda ts: agg[str(ts)]["mean"])
    speedup = (baseline_16 / agg[str(best_ts)]["mean"] - 1) * 100 if agg[str(best_ts)]["mean"] > 0 else 0

    out = {
        "schema_version": 1, "phase": "C asymmetric tile policy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_values": TILE_VALS,
                      "cameras": list(range(a.camera_offset, a.camera_offset + a.camera_count))},
        "camera_records": records, "aggregate": agg,
        "baseline_tile_16_Titer_ms": baseline_16,
        "best_fixed_tile": best_ts, "best_tile_Titer_ms": agg[str(best_ts)]["mean"],
        "speedup_over_baseline_pct": speedup,
        "verdict": "MAYBE"
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}  best_tile={best_ts}  speedup={speedup:.1f}%")

if __name__ == "__main__":
    main()
