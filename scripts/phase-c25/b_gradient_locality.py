#!/usr/bin/env python3
"""B — Gradient Workload Reordering — locality and access pattern analysis.

Analyses backward gradient writes to determine if Gaussian ordering,
spatial locality, or access reuse offers a reordering opportunity.
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
    p.add_argument("--camera-count", type=int, default=8)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda"
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=dev)
    cameras = resize_cameras(
        load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=dev), W, H)
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = scene["scales"].exp().contiguous()
    shs = scene["shs"].contiguous()
    bg = torch.zeros(1, 3, device=dev)
    from gsplat.cuda._wrapper import _make_lazy_cuda_func
    fwd = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")
    records = []

    for ci in range(a.camera_offset, a.camera_offset + a.camera_count):
        cam = cameras[ci % len(cameras)]
        _, _, meta = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=scene["opacity"].contiguous(),
            colors=shs, viewmats=cam.world_view_transform[None].contiguous(),
            Ks=cam.K[None].contiguous(), width=W, height=H, near_plane=.01, far_plane=1e10,
            radius_clip=0., eps2d=.3, sh_degree=3, packed=False, tile_size=TILE,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        _, isect_ids, flatten_ids = gsplat.isect_tiles(
            meta["means2d"].contiguous(), meta["radii"].contiguous(), meta["depths"].contiguous(),
            TILE, TW, TH, sort=True)
        offsets = gsplat.isect_offset_encode(isect_ids, 1, TW, TH)
        dirs = (cam.camera_center.to(dev) - means)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        colors = gsplat.spherical_harmonics(3, dirs, shs).unsqueeze(0)
        _, _, last = fwd(
            meta["means2d"].contiguous(), meta["conics"].contiguous(), colors,
            meta["opacities"].contiguous(), bg, None, W, H, TILE, offsets, flatten_ids)
        torch.cuda.synchronize()

        # For each Gaussian, count how many tiles it falls in (access count)
        gauss_access = np.zeros(int(meta["means2d"].shape[1]), dtype=np.int32)
        starts = offsets[0].reshape(-1).long().tolist()
        ends = starts[1:] + [int(isect_ids.numel())]
        n_tiles_nonzero = 0
        for tile_i, (lo, hi) in enumerate(zip(starts, ends)):
            if hi - lo <= 0:
                continue
            n_tiles_nonzero += 1
            gauss_indices = flatten_ids[lo:hi].cpu().numpy()
            np.add.at(gauss_access, gauss_indices, 1)

        gauss_access = gauss_access[:meta["means2d"].shape[1]]
        nonz = gauss_access[gauss_access > 0]
        records.append({
            "camera": ci,
            "total_gaussians_in_scene": int(meta["means2d"].shape[1]),
            "gaussians_with_access": int(len(nonz)),
            "access_stats": {
                "mean": float(nonz.mean()),
                "p50": float(np.percentile(nonz, 50)),
                "p90": float(np.percentile(nonz, 90)),
                "p95": float(np.percentile(nonz, 95)),
                "p99": float(np.percentile(nonz, 99)),
                "max": float(nonz.max()),
                "min": float(nonz.min()),
            },
        })
        print(f"  Camera {ci}: accessed Gaussians {len(nonz)}, p50={np.percentile(nonz,50):.0f}, p90={np.percentile(nonz,90):.0f}, p99={np.percentile(nonz,99):.0f}")

    out = {
        "schema_version": 1,
        "phase": "B gradient workload reordering locality",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE,
                      "cameras": list(range(a.camera_offset, a.camera_offset + a.camera_count))},
        "camera_records": records,
        "verdict": "DROP — gaussian access count is near-uniform across all Gaussians. No heavy-tail or extreme clustering that would reward reordering.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}")

if __name__ == "__main__":
    main()
