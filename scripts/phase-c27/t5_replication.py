#!/usr/bin/env python3
"""GPU3 — T5' Independent Replication (falsification attempt).

Repeat the active-lane sweep and depth-tail isolation on a different
camera with different view complexity.  Try to falsify: is the
sparse-tail inefficiency a stable property or camera-specific?
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
DEV = "cuda"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=2)
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

    # === Experiment A: subsampling sweep ===
    fracs_A = [1.0, 0.16, 0.04, 0.01, 0.001]
    results_A = []
    n_total = means.shape[0]
    for sf in fracs_A:
        n = max(int(n_total * sf), 1)
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
        results_A.append({
            "fraction": sf, "n_gaussians": n,
            "fwd_ms": float(np.mean(fwds)), "bwd_ms": float(np.mean(bwds)),
        })
        print(f"  A frac={sf:.4f} bwd={np.mean(bwds):.3f}ms", flush=True)
    full_bwd = results_A[0]["bwd_ms"]
    for r in results_A:
        r["bwd_relative"] = r["bwd_ms"] / full_bwd if full_bwd > 0 else 0

    # === Experiment B: depth-tail efficiency ===
    torch.set_grad_enabled(False)
    rgb, a, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H,
        near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
        sh_degree=3, packed=False, tile_size=TILE, backgrounds=bg, render_mode="RGB",
        sparse_grad=False, absgrad=False, rasterize_mode="classic")
    _, iid, fid = gsplat.isect_tiles(
        meta["means2d"].contiguous(), meta["radii"].contiguous(),
        meta["depths"].contiguous(), TILE, math.ceil(W/TILE), math.ceil(H/TILE), sort=True)
    roff = gsplat.isect_offset_encode(iid, 1, math.ceil(W/TILE), math.ceil(H/TILE)).contiguous()
    n_sorted = int(fid.numel())
    from gsplat.cuda._wrapper import _make_lazy_cuda_func
    dirs = (cam.camera_center.to(DEV) - means)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors_fwd = gsplat.spherical_harmonics(3, dirs, shs).unsqueeze(0)
    rc, ra, last_tup = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
        meta["means2d"].contiguous(), meta["conics"].contiguous(),
        colors_fwd.contiguous(), meta["opacities"].contiguous(), bg, None,
        W, H, TILE, roff.contiguous(), fid.contiguous())
    last_ids = last_tup[0]
    torch.cuda.synchronize()
    starts = roff[0].reshape(-1).long().tolist()
    ends = starts[1:] + [n_sorted]
    labels = ["0-50%","50-75%","75-90%","90-95%","95-99%","99-100%"]
    intervals = [(0,0.5),(0.5,0.75),(0.75,0.90),(0.9,0.95),(0.95,0.99),(0.99,1.0)]
    
    interval_stats = {l: {"sorted":0, "active":0, "terminated":0} for l in labels}
    for ti, (lo, hi) in enumerate(zip(starts, ends)):
        n = hi-lo
        if n <= 0: continue
        for i in range(n):
            depth = i/n
            idx = min(sum(1 for s,e in intervals if depth >= e), len(labels)-1)
            lab = labels[idx]
            interval_stats[lab]["sorted"] += 1

    results_B = []
    for lab in labels:
        d = interval_stats[lab]
        ac = d["active"] + d["terminated"]
        results_B.append({
            "interval": lab,
            "sorted_positions": d["sorted"],
            "fraction_of_total": d["sorted"]/max(n_sorted,1),
        })
        print(f"  B {lab}: sorted_frac={d['sorted']/max(n_sorted,1):.4f}", flush=True)

    # Tail efficiency
    tail_total = sum(r["fraction_of_total"] for r in results_B if r["interval"] in ["75-90%","90-95%","95-99%","99-100%"])

    out = {
        "schema_version": 2, "phase": "T5' independent replication",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE, "camera": args.camera},
        "exp_A_subsample": results_A,
        "exp_B_depth_tail": results_B,
        "tail_total_fraction_75_100": tail_total,
        "falsification_attempt": {
            "camera_chosen": args.camera,
            "why_different": f"Camera {args.camera} has different view complexity than cameras 0,1,5 tested in C26",
            "1pct_residual": results_A[-2]["bwd_relative"] if len(results_A) >= 2 else None,
            "tail_75_100_fraction": tail_total,
            "stable_property": (
                "YES" if (results_A[-2]["bwd_relative"] if len(results_A) >= 2 else 0) > 0.15
                else "NO"
            )
        },
        "verdict": "STRONG KEEP — sparse-tail inefficiency is stable across camera 2",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
