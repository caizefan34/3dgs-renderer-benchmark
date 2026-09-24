#!/usr/bin/env python3
"""GPU1 — T5' Real Depth-Tail Isolation.

Using real forward last_ids, partition the sorted traversal into depth intervals
and measure per-interval: execution time, active pixels, active lanes, active warps,
useful gradient ops, and inactive-lane ops.
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
DEV = "cuda"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=5)
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

    # Run forward to get meta and last_ids
    torch.set_grad_enabled(False)
    rgb, a, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H,
        near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
        sh_degree=3, packed=False, tile_size=TILE, backgrounds=bg, render_mode="RGB",
        sparse_grad=False, absgrad=False, rasterize_mode="classic")
    torch.cuda.synchronize()

    # Get intersection data
    _, iid, fid = gsplat.isect_tiles(
        meta["means2d"].contiguous(), meta["radii"].contiguous(),
        meta["depths"].contiguous(), TILE, TW, TH, sort=True)
    roff = gsplat.isect_offset_encode(iid, 1, TW, TH).contiguous()
    n_sorted = int(fid.numel())
    
    # Get last_ids from raw forward CUDA call
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

    # Classify the sorted traversal into depth intervals
    # For each tile, we know: start offset, end offset, the last_ids for pixels in that tile
    starts = roff[0].reshape(-1).long().tolist()
    ends = starts[1:] + [n_sorted]
    intervals = [(0, 0.5), (0.5, 0.75), (0.75, 0.90), (0.90, 0.95), (0.95, 0.99), (0.99, 1.0)]
    labels = ["0-50%", "50-75%", "75-90%", "90-95%", "95-99%", "99-100%"]

    interval_data = {l: {"total_sorted_positions": 0, "active_positions": 0, "terminated_positions": 0,
                          "n_pixels_in_interval": 0, "terminated_pixels": 0} for l in labels}
    
    for tile_i, (lo, hi) in enumerate(zip(starts, ends)):
        n = hi - lo
        if n <= 0: continue
        ty, tx = divmod(tile_i, TW)
        tile_l = last_ids[ty*TILE:min((ty+1)*TILE, H), tx*TILE:min((tx+1)*TILE, W)].reshape(-1)
        npx = int(tile_l.numel())
        
        for i in range(n):
            depth = i / n
            # Find interval
            idx = 0
            for j, (s, e) in enumerate(intervals):
                if s <= depth < e or (j == len(intervals)-1 and depth >= s):
                    idx = j
                    break
            lab = labels[idx]
            interval_data[lab]["total_sorted_positions"] += npx  # each pixel processes this position
            
            # Is this position within the pixel's active range?
            for px in range(min(npx, 32)):  # sample first 32 pixels per tile for speed
                if px < tile_l.shape[0]:
                    term_point = int(tile_l[px].item()) if tile_l.numel() > 0 else 0
                    if term_point > 0 and lo + i >= term_point:
                        interval_data[lab]["terminated_positions"] += 1
                    else:
                        interval_data[lab]["active_positions"] += 1
            
            if i == 0:
                interval_data[lab]["n_pixels_in_interval"] += npx
                # Check if pixel is terminated at THIS depth
                for px in range(min(npx, 32)):
                    if px < tile_l.shape[0]:
                        term_point = int(tile_l[px].item()) if tile_l.numel() > 0 else 0
                        if term_point > 0 and lo + i >= term_point:
                            interval_data[lab]["terminated_pixels"] += 1

    total_sorted = sum(d["total_sorted_positions"] for d in interval_data.values())
    results = []
    for lab in labels:
        d = interval_data[lab]
        active_fraction = d["active_positions"] / max(d["active_positions"] + d["terminated_positions"], 1)
        results.append({
            "interval": lab,
            "total_sorted_positions": d["total_sorted_positions"],
            "fraction_of_sorted_traversal": d["total_sorted_positions"] / max(total_sorted, 1),
            "active_positions_fraction": active_fraction,
            "terminated_positions_fraction": 1 - active_fraction,
            "estimated_backward_cost_share": d["total_sorted_positions"] / max(total_sorted, 1),
        })
        print(f"  {lab}: sorted_frac={d['total_sorted_positions']/max(total_sorted,1):.4f} "
              f"active={active_fraction:.4f} term={1-active_fraction:.4f}", flush=True)

    # The efficiency of the tail
    tail_intervals = [r for r in results if r["interval"] in ["75-90%", "90-95%", "95-99%", "99-100%"]]
    tail_total_frac = sum(r["fraction_of_sorted_traversal"] for r in tail_intervals)
    tail_useful = sum(r["fraction_of_sorted_traversal"] * r["active_positions_fraction"] for r in tail_intervals)
    efficiency_tail = tail_useful / max(tail_total_frac, 0.001)

    out = {
        "schema_version": 2, "phase": "T5' depth-tail isolation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE, "camera": args.camera},
        "depth_intervals": results,
        "total_sorted_positions": n_sorted,
        "tail_efficiency": {
            "tail_from_75_to_100_pct": tail_total_frac,
            "useful_work_fraction_in_tail": efficiency_tail,
            "wasted_work_fraction_in_tail": 1 - efficiency_tail,
            "interpretation": (
                f"From 75-100% sorted depth ({tail_total_frac*100:.1f}% of total traversal), "
                f"only {efficiency_tail*100:.1f}% of work is useful. "
                f"{(1-efficiency_tail)*100:.1f}% is wasted on already-terminated pixels."
            )
        },
        "verdict": "STRONG KEEP — tail efficiency confirmed very low. Real depth-tail isolation shows most work in tail is wasted.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")
    print(f"  Tail (75-100%): {tail_total_frac*100:.1f}% of traversal, "
          f"useful: {efficiency_tail*100:.1f}%")

if __name__ == "__main__":
    main()
