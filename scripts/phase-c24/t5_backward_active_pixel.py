#!/usr/bin/env python3
"""T5 — Backward Active-Pixel / Activity-Aware Execution.

Determine whether forward termination information can safely constrain
backward execution.  For each pixel, compare the forward endpoint
(last_ids) with the full tile range.  Measure how many pixels are
already terminated at each depth fraction, and estimate the backward
work that would be on already-terminated pixels.

DO NOT equate low active-pixel count with 100% removable work.
Backward must still handle gradient accumulation for the active
pixels that DO have contributions.
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
    p.add_argument("--camera-count", type=int, default=12)
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
        last_ids = last[0].long()
        starts = offsets[0].reshape(-1).long().tolist()
        ends = starts[1:] + [int(isect_ids.numel())]

        total_pixels = 0
        active_pixel_steps_all = 0  # total backward steps on active pixels
        terminated_pixel_steps_suffix = 0  # steps on already-terminated pixels
        total_range_steps = 0

        for tile_i, (lo, hi) in enumerate(zip(starts, ends)):
            n = hi - lo
            if n <= 0:
                continue
            ty, tx = divmod(tile_i, TW)
            tile_last = last_ids[ty*TILE:min((ty+1)*TILE, H), tx*TILE:min((tx+1)*TILE, W)]
            npx = int(tile_last.numel())
            total_pixels += npx
            total_range_steps += n * npx

            # For each pixel: forward stopped at last_ids.  In backward,
            # the same range [0..last_ids] must be traversed (at minimum).
            # Steps after last_ids are on pixels already terminated in forward.
            prefix = (tile_last.reshape(-1) - lo + 1).clamp(0, n).long()
            active_pixel_steps_all += int(prefix.sum().item())
            suffix = n - prefix
            terminated_pixel_steps_suffix += int(suffix.sum().item())

        cam_rec = {
            "camera": ci,
            "total_pixels": total_pixels,
            "total_range_steps_if_naive": total_range_steps,
            "active_pixel_steps_after_forward_termination": active_pixel_steps_all,
            "terminated_pixel_steps_suffix": terminated_pixel_steps_suffix,
            "fraction_suffix_on_terminated_pixels": terminated_pixel_steps_suffix / max(total_range_steps, 1),
            "fwd_endpoint_active_fraction": active_pixel_steps_all / max(total_range_steps, 1),
        }
        records.append(cam_rec)
        print(f"  Camera {ci}: active_frac={cam_rec['fwd_endpoint_active_fraction']:.4f}  "
              f"terminated_suffix_frac={cam_rec['fraction_suffix_on_terminated_pixels']:.4f}", flush=True)

    fracs = np.array([r["fraction_suffix_on_terminated_pixels"] for r in records])
    out = {
        "schema_version": 1,
        "phase": "T5 backward active-pixel feasibility",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE,
                      "cameras": list(range(a.camera_offset, a.camera_offset + a.camera_count))},
        "camera_records": records,
        "cross_camera_aggregate": {
            "terminated_suffix_fraction_mean": float(fracs.mean()),
            "terminated_suffix_fraction_min": float(fracs.min()),
            "terminated_suffix_fraction_max": float(fracs.max()),
        },
        "interpretation": "The terminated-pixel suffix fraction represents the portion of per-pixel range steps that occur on pixels already finished in forward. If backward skips these entirely (compaction), the maximum possible reduction is this fraction of backward traversal. However: (1) backward must still accumulate gradients contributed by the active portion, and (2) skipping requires per-pixel active masks that themselves have overhead.",
        "verdict": (
            "KEEP_CANDIDATE" if fracs.mean() > 0.50
            else "MAYBE" if fracs.mean() > 0.20
            else "DROP"
        ),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}")
    print(f"Terminated suffix fraction mean={fracs.mean():.4f}")

if __name__ == "__main__":
    main()
