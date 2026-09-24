#!/usr/bin/env python3
"""T5' — Sparse-Tail Backward Reorganization.

Measure backward kernel time as a function of active-lane fraction
across cameras with varying view complexity.

Active fraction is computed per camera from forward last_ids.
Key graph: T_backward = f(active_fraction).
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

def compute_active_frac_pixels(m2d, conics, colors, opacities, bg, roff, fid):
    """Compute per-pixel active fraction using forward last_ids."""
    from gsplat.cuda._wrapper import _make_lazy_cuda_func
    rc, ra, last = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
        m2d.contiguous(), conics.contiguous(), colors.contiguous(),
        opacities.contiguous(), bg, None, W, H, TILE, roff.contiguous(), fid.contiguous())
    last_ids = last[0]
    starts = roff[0].reshape(-1).long().tolist()
    n_sorted = int(fid.numel())
    ends = starts[1:] + [n_sorted]
    total_pixels = 0
    total_active_steps = 0
    for tile_i, (lo, hi) in enumerate(zip(starts, ends)):
        n = hi - lo
        if n <= 0: continue
        ty, tx = divmod(tile_i, TW)
        tile_l = last_ids[ty*TILE:min((ty+1)*TILE, H), tx*TILE:min((tx+1)*TILE, W)]
        npx = int(tile_l.numel())
        total_pixels += npx
        prefix = (tile_l.reshape(-1) - lo + 1).clamp(0, n).long()
        total_active_steps += int(prefix.float().mean().item()) * npx
    return total_active_steps / max(total_pixels, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera-offset", type=int, default=0)
    p.add_argument("--camera-count", type=int, default=8)
    args = p.parse_args()
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

    records = []

    for ci in range(args.camera_offset, args.camera_offset + args.camera_count):
        cam = cameras[ci % len(cameras)]
        dirs = (cam.camera_center.to(dev) - means)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        colors = gsplat.spherical_harmonics(3, dirs, shs).unsqueeze(0)

        # Full forward+backward using autograd-enabled gsplat.rasterization
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
        fwd_ms = (t1 - t0) * 1000
        bwd_ms = (t2 - t1) * 1000

        # Compute intersection data for active fraction
        _, iid, fid = gsplat.isect_tiles(
            meta["means2d"].contiguous(), meta["radii"].contiguous(),
            meta["depths"].contiguous(), TILE, TW, TH, sort=True)
        roff = gsplat.isect_offset_encode(iid, 1, TW, TH).contiguous()

        # Active fraction from forward
        active_frac = compute_active_frac_pixels(
            meta["means2d"].contiguous(), meta["conics"].contiguous(),
            colors.contiguous(), meta["opacities"].contiguous(), bg, roff, fid.contiguous())

        records.append({
            "camera": ci,
            "active_fraction": active_frac,
            "fwd_ms": fwd_ms,
            "bwd_ms": bwd_ms,
            "T_iter_ms": fwd_ms + bwd_ms,
        })
        print(f"  Camera {ci}: active_frac={active_frac:.4f}  fwd={fwd_ms:.3f}ms  bwd={bwd_ms:.3f}ms", flush=True)

    fracs = np.array([r["active_fraction"] for r in records])
    bwd = np.array([r["bwd_ms"] for r in records])
    corr = float(np.corrcoef(fracs, bwd)[0, 1]) if len(fracs) > 1 else 0

    out = {
        "schema_version": 1,
        "phase": "T5' sparse-tail backward reorganization",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE,
                      "cameras": list(range(args.camera_offset, args.camera_offset + args.camera_count))},
        "camera_records": records,
        "active_fraction_vs_backward_time_correlation": corr,
        "active_frac_range": [float(fracs.min()), float(fracs.max())],
        "bwd_range_ms": [float(bwd.min()), float(bwd.max())],
        "evidence": (
            f"Active fraction range: [{fracs.min():.4f}, {fracs.max():.4f}]. "
            f"Backward time range: [{bwd.min():.3f}, {bwd.max():.3f}]ms. "
            f"Correlation: {corr:.4f}. "
            f"High-positive => backward time scales with active fraction, kernel handles sparsity well. "
            f"Weak/flat => sparse tail imposes disproportionate cost."
        ),
        "verdict": "MAYBE — pending correlation analysis."
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")
    print(f"  Active fraction range: [{fracs.min():.4f}, {fracs.max():.4f}]")
    print(f"  Backward time range: [{bwd.min():.3f}, {bwd.max():.3f}]ms")
    print(f"  Correlation: {corr:.4f}")

if __name__ == "__main__":
    main()
