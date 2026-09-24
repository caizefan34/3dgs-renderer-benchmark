#!/usr/bin/env python3
"""C23 observational activity-decay measurement using gsplat's real last_ids."""
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
FRACTIONS = np.linspace(0.0, 0.95, 20)

def stat(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "min": float(a.min()), "max": float(a.max())}

def classify_quantile(n, p50, p90):
    """Use 4× p50 as HIGH threshold, 2× p50 as MEDIUM, else LOW."""
    if n >= 4 * p50: return "HIGH"
    if n >= 2 * p50: return "MEDIUM"
    return "LOW"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera-offset", type=int, required=True)
    p.add_argument("--camera-count", type=int, default=12)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device="cuda")
    cameras = resize_cameras(load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device="cuda"), W, H)
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = scene["scales"].exp().contiguous()
    shs = scene["shs"].contiguous()
    bg = torch.zeros(1, 3, device="cuda")
    from gsplat.cuda._wrapper import _make_lazy_cuda_func
    fwd = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")
    records = []

    for ci in range(a.camera_offset, a.camera_offset + a.camera_count):
        cam = cameras[ci]
        _, _, meta = gsplat.rasterization(means=means, quats=quats, scales=scales, opacities=scene["opacity"].contiguous(), colors=shs, viewmats=cam.world_view_transform[None].contiguous(), Ks=cam.K[None].contiguous(), width=W, height=H, near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3, sh_degree=3, packed=False, tile_size=TILE, backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False, rasterize_mode="classic")
        _, isect_ids, flatten_ids = gsplat.isect_tiles(meta["means2d"].contiguous(), meta["radii"].contiguous(), meta["depths"].contiguous(), TILE, TW, TH, sort=True)
        offsets = gsplat.isect_offset_encode(isect_ids, 1, TW, TH)
        dirs = cam.camera_center.cuda() - means
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        colors = gsplat.spherical_harmonics(3, dirs, shs).unsqueeze(0)
        _, _, last_ids = fwd(meta["means2d"].contiguous(), meta["conics"].contiguous(), colors, meta["opacities"].contiguous(), bg, None, W, H, TILE, offsets, flatten_ids)
        torch.cuda.synchronize()
        last_ids = last_ids[0].long()
        starts = offsets[0].reshape(-1).long().tolist()
        ends = starts[1:] + [int(isect_ids.numel())]
        groups = {g: {"tile_records": []} for g in ("LOW", "MEDIUM", "HIGH")}

        all_lens = [hi2 - lo2 for lo2, hi2 in zip(starts, ends) if (hi2 - lo2) > 0]
        if not all_lens:
            continue
        p50v = float(np.percentile(all_lens, 50))
        p90v = float(np.percentile(all_lens, 90))

        for tile_i, (lo, hi) in enumerate(zip(starts, ends)):
            n = hi - lo
            if not n:
                continue
            ty, tx = divmod(tile_i, TW)
            endpoints = (last_ids[ty*TILE:min((ty+1)*TILE,H), tx*TILE:min((tx+1)*TILE,W)].reshape(-1) - lo + 1).clamp(0, n)
            active_curve = []
            warp_curve = []
            padded = torch.nn.functional.pad(endpoints, (0, (-endpoints.numel()) % 32), value=0).reshape(-1, 32)
            for frac in FRACTIONS:
                k = int(frac * n)
                active_curve.append(float((endpoints > k).float().mean().item()))
                warp_curve.append(float((padded > k).float().mean().item()))
            tail_lane_fraction = float((padded > int(.9*n)).float().mean().item())
            low_util_warps = float(((padded > int(.9*n)).float().mean(dim=1) <= .125).float().mean().item())
            groups[classify_quantile(n, p50v, p90v)]["tile_records"].append({"tile": tile_i, "total_intersections": n, "pixel_count": int(endpoints.numel()), "activity_curve": active_curve, "warp_active_lane_curve": warp_curve, "mean_activity": float(np.mean(active_curve)), "activity_at_50pct": active_curve[10], "activity_at_90pct": active_curve[18], "active_lane_fraction_at_90pct": tail_lane_fraction, "warps_le_4_active_lanes_at_90pct": low_util_warps})

        result_groups = {}
        for g, data in groups.items():
            tiles = data["tile_records"]
            result_groups[g] = {"n_tiles": len(tiles), "mean_activity": stat([t["mean_activity"] for t in tiles]) if tiles else None, "activity_at_90pct": stat([t["activity_at_90pct"] for t in tiles]) if tiles else None, "active_lane_fraction_at_90pct": stat([t["active_lane_fraction_at_90pct"] for t in tiles]) if tiles else None, "warps_le_4_active_lanes_at_90pct": stat([t["warps_le_4_active_lanes_at_90pct"] for t in tiles]) if tiles else None, "tile_records": tiles}
        records.append({"camera": ci, "density_groups": result_groups})
        print("camera", ci, "high", result_groups["HIGH"]["n_tiles"], flush=True)

    high_tiles = [t for r in records for t in r["density_groups"]["HIGH"]["tile_records"]]
    high_aggregate = None
    if high_tiles:
        high_aggregate = {
            "mean_activity": stat([t["mean_activity"] for t in high_tiles]),
            "activity_at_90pct": stat([t["activity_at_90pct"] for t in high_tiles]),
            "active_lane_fraction_at_90pct": stat([t["active_lane_fraction_at_90pct"] for t in high_tiles]),
            "warps_le_4_active_lanes_at_90pct": stat([t["warps_le_4_active_lanes_at_90pct"] for t in high_tiles]),
        }
    out = {
        "schema_version": 1,
        "phase": "C23 activity decay",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instrumentation": "Direct existing CUDA forward call captures last_ids; no rasterization behavior is changed. Activity is reconstructed from per-pixel endpoints. Warp mapping is a row-major groups-of-32 utilization proxy, not a hardware counter.",
        "semantics": "A(k) is the fraction of pixel lanes whose last committed endpoint is later than sorted-position k. This describes baseline early-termination activity decay and does not itself prove removable work.",
        "protocol": {"scene": "Mip-NeRF360 room", "resolution": "1920x1080", "tile_size": TILE, "cameras": list(range(a.camera_offset, a.camera_offset+a.camera_count))},
        "camera_records": records,
        "high_density_aggregate": high_aggregate,
        "decision": "MAYBE",
        "decision_basis": "Activity decay is direct evidence, but incremental dense-CTA execution cost needs hardware counters or a controlled alternate execution experiment; neither is performed here.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)

if __name__ == "__main__":
    main()
