#!/usr/bin/env python3
"""C20 reproducible gsplat benchmark: tile/CTA configuration or camera workload sweep.

The standard gsplat backend couples logical tile size to rasterizer CTA geometry.
Therefore a tile sweep measures a configuration effect, not an execution-decoupled
CTA mechanism. Results include per-camera workload descriptors and pixel checks.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras


def event_ms(fn, repeats: int) -> list[float]:
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def summary(samples: list[float]) -> dict:
    a = np.asarray(samples, dtype=np.float64)
    return {"samples_ms": [round(float(x), 6) for x in samples], "mean_ms": round(float(a.mean()), 6),
            "median_ms": round(float(np.median(a)), 6), "std_ms": round(float(a.std()), 6),
            "p95_ms": round(float(np.percentile(a, 95)), 6), "p99_ms": round(float(np.percentile(a, 99)), 6)}


def payload(scene, cam, width, height, tile_size):
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = scene["scales"].exp().contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    shs = scene["shs"].contiguous()
    bg = torch.zeros(1, 3, device="cuda")
    kwargs = dict(means=means, quats=quats, scales=scales, opacities=opacities, colors=shs,
                  viewmats=cam.world_view_transform.unsqueeze(0).contiguous(), Ks=cam.K.unsqueeze(0).contiguous(),
                  width=width, height=height, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
                  eps2d=0.3, sh_degree=3, packed=False, tile_size=tile_size, backgrounds=bg,
                  render_mode="RGB", sparse_grad=False, absgrad=False, rasterize_mode="classic")
    return kwargs


def tile_load(meta, tile_size, width, height):
    ids = meta["isect_ids"]
    n_tiles = math.ceil(width / tile_size) * math.ceil(height / tile_size)
    # gsplat key uses low tile bits after depth; modulo n_tiles is the established C19 decoder.
    counts = torch.bincount((ids % n_tiles).long(), minlength=n_tiles).cpu().numpy()
    return {"total_intersections": int(ids.numel()), "active_tiles": int((counts > 0).sum()),
            "total_tiles": int(n_tiles), "mean_intersections_per_tile": round(float(counts.mean()), 4),
            "p50_intersections_per_tile": int(np.percentile(counts, 50)),
            "p90_intersections_per_tile": int(np.percentile(counts, 90)),
            "p99_intersections_per_tile": int(np.percentile(counts, 99)), "max_intersections_per_tile": int(counts.max()),
            "visible_gaussians": int((meta["radii"].squeeze() > 0).sum().item())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tiles", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32])
    p.add_argument("--camera-count", type=int, default=8)
    p.add_argument("--camera-offset", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--width", type=int, default=1920); p.add_argument("--height", type=int, default=1080)
    p.add_argument("--candidate", default="C2")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device="cuda")
    cameras = resize_cameras(load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device="cuda"), args.width, args.height)
    selected = [cameras[(args.camera_offset + i) % len(cameras)] for i in range(args.camera_count)]
    results = []
    reference = None
    for tile in args.tiles:
        per_camera = []
        for camera_index, cam in enumerate(selected):
            kwargs = payload(scene, cam, args.width, args.height, tile)
            for _ in range(args.warmup): gsplat.rasterization(**kwargs)
            torch.cuda.synchronize()
            output, alpha, meta = gsplat.rasterization(**kwargs)
            torch.cuda.synchronize()
            timing = summary(event_ms(lambda: gsplat.rasterization(**kwargs), args.repeats))
            load = tile_load(meta, tile, args.width, args.height)
            rec = {"camera_index": args.camera_offset + camera_index, "timing": timing, "workload": load}
            if reference is None and tile == 16:
                reference = output.detach().clone()
                rec["correctness_vs_tile16"] = {"reference": True, "max_abs": 0.0, "mse": 0.0, "psnr_db": "inf", "pass": True}
            elif reference is not None:
                mse = float(torch.mean((output-reference).square()).item()); max_abs = float(torch.max(torch.abs(output-reference)).item())
                psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
                rec["correctness_vs_tile16"] = {"reference": False, "max_abs": max_abs, "mse": mse, "psnr_db": psnr, "pass": psnr >= 45.0}
            per_camera.append(rec)
        medians = [x["timing"]["median_ms"] for x in per_camera]
        results.append({"tile_size": tile, "cta_shape": f"{tile}x{tile}x1", "threads_per_block": tile*tile,
                        "dynamic_shared_memory_bytes_estimate": tile*tile*28,
                        "per_camera": per_camera, "aggregate": summary(medians)})
    out = {"schema_version": 1, "candidate": args.candidate, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "gsplat": gsplat.__version__},
           "protocol": {"scene": "room official Mip-NeRF360", "gaussians": int(scene["num_points"]), "resolution": f"{args.width}x{args.height}", "packed": False, "cameras": [args.camera_offset+i for i in range(args.camera_count)], "warmup": args.warmup, "repeats": args.repeats},
           "interpretation_guard": "Standard gsplat couples logical tile size and rasterizer CTA geometry. This sweep establishes configuration gain only; it does not establish a decoupled-CTA mechanism gain.", "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"out": args.out, "configs": len(results)}, indent=2))

if __name__ == "__main__": main()
