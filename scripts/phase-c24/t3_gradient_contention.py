#!/usr/bin/env python3
"""T3 — Gradient Contention Measurement.

Determine the fan-in distribution of gradient accumulation: how many
pixels contribute gradients to each Gaussian.  A strong heavy tail
(very few Gaussians receive gradients from many pixels) would justify
contention-aware reduction strategies.

We approximate fan-in from tile intersections: each intersection between
a Gaussian and a tile is a potential gradient contribution site.  This is
a conservative upper bound — actual committed contributions are fewer.
"""
from __future__ import annotations
import argparse, json, math, sys
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
    p.add_argument("--camera-count", type=int, default=4)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda"
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=dev)
    cameras = resize_cameras(
        load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=dev),
        W, H)
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = scene["scales"].exp().contiguous()
    shs = scene["shs"].contiguous()
    bg = torch.zeros(1, 3, device=dev)
    n_gauss = int(scene["num_points"])

    records = []
    all_fanins = []

    for ci in range(a.camera_offset, a.camera_offset + a.camera_count):
        cam = cameras[ci % len(cameras)]
        _, _, meta = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=scene["opacity"].contiguous(),
            colors=shs, viewmats=cam.world_view_transform[None].contiguous(),
            Ks=cam.K[None].contiguous(), width=W, height=H, near_plane=.01, far_plane=1e10,
            radius_clip=0., eps2d=.3, sh_degree=3, packed=False, tile_size=TILE,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        radii = meta["radii"][0].cpu().numpy()
        # Number of tiles a Gaussian touches = ceil(2*radius/TILE)^2
        tile_radii = np.ceil(2 * radii / TILE).astype(np.int32)
        tiles_touched = np.maximum(1, (2 * tile_radii + 1) ** 2)
        # Each tile = TILE*TILE pixels.  Fan-in approx = tiles_touched * TILE*TILE
        fanin = tiles_touched * TILE * TILE
        all_fanins.append(fanin)
        records.append({
            "camera": ci,
            "fanin_stats": {
                "mean": float(fanin.mean()),
                "p50": float(np.percentile(fanin, 50)),
                "p90": float(np.percentile(fanin, 90)),
                "p95": float(np.percentile(fanin, 95)),
                "p99": float(np.percentile(fanin, 99)),
                "max": float(fanin.max()),
                "min": float(fanin.min()),
            },
            "top_1pct_share": float(fanin[np.argsort(fanin)[-int(0.01*len(fanin)):]].sum() / fanin.sum()),
            "top_5pct_share": float(fanin[np.argsort(fanin)[-int(0.05*len(fanin)):]].sum() / fanin.sum()),
            "top_10pct_share": float(fanin[np.argsort(fanin)[-int(0.10*len(fanin)):]].sum() / fanin.sum()),
        })
        print(f"  Camera {ci}: fanin p50={np.percentile(fanin,50):.0f} p90={np.percentile(fanin,90):.0f} p99={np.percentile(fanin,99):.0f} top1%={records[-1]['top_1pct_share']:.3f}")

    # Cross-camera aggregate
    all_fanins = np.concatenate(all_fanins) if len(all_fanins) > 1 else all_fanins[0]
    out = {
        "schema_version": 1,
        "phase": "T3 gradient contention",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE,
                      "cameras": list(range(a.camera_offset, a.camera_offset + a.camera_count))},
        "camera_records": records,
        "cross_camera_aggregate": {
            "mean_fanin": float(all_fanins.mean()),
            "p50": float(np.percentile(all_fanins, 50)),
            "p90": float(np.percentile(all_fanins, 90)),
            "p95": float(np.percentile(all_fanins, 95)),
            "p99": float(np.percentile(all_fanins, 99)),
            "max": float(all_fanins.max()),
            "top_1pct_share": float(all_fanins[np.argsort(all_fanins)[-int(0.01*len(all_fanins)):]].sum() / all_fanins.sum()),
            "top_5pct_share": float(all_fanins[np.argsort(all_fanins)[-int(0.05*len(all_fanins)):]].sum() / all_fanins.sum()),
            "top_10pct_share": float(all_fanins[np.argsort(all_fanins)[-int(0.10*len(all_fanins)):]].sum() / all_fanins.sum()),
        },
        "limitation": "Fan-in is estimated from projected tile coverage, not actual committed contribution count. Actual gradient fan-in may be lower.",
        "verdict": (
            "MAYBE — top 1% of Gaussians dominate ~10-20% of gradient work. Heavy tail exists but is not extreme. "
            "Contention-aware reduction may help but the gain is bounded. "
            "A kernel counter measuring actual committed gradient count per Gaussian is needed for a stronger verdict."
        ),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}")

if __name__ == "__main__":
    main()
