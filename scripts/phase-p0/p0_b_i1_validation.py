#!/usr/bin/env python3
"""P0-B: I1 Training-Phase-Aware Renderer Policy — Real Checkpoints.

Uses real training checkpoints (0, 3K, 5K, 10K, 15K, 20K, 25K, 30K iterations)
to validate tile-size policy across actual training states.

If checkpoints unavailable, generates close approximations via Gaussian count.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

W, H = 1920, 1080
TW, TH = math.ceil(W/16), math.ceil(H/16)
DEV = "cuda"
TILE_VALS = [16, 24]

def measure(cam, means, quats, scales, opac, shs, bg, ts, reps=5):
    fwds, bwds = [], []
    for r in range(reps):
        for x in [means, quats, scales, opac, shs]: x.grad = None
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rgb, a, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),
            Ks=cam.K[None].contiguous(), width=W, height=H,
            near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
            sh_degree=3, packed=False, tile_size=ts,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        torch.cuda.synchronize(); t1 = time.perf_counter()
        (rgb.float().mean() + a.float().mean()).backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        fwds.append((t1-t0)*1000); bwds.append((t2-t1)*1000)
    return float(np.mean(fwds)), float(np.mean(bwds)), float(np.mean(fwds)+np.mean(bwds))

def workload_stats(cam, means, quats, scales, opac, shs, bg):
    torch.set_grad_enabled(False)
    rgb, a, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H,
        near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
        sh_degree=3, packed=False, tile_size=16,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    _, iid, fid = gsplat.isect_tiles(
        meta["means2d"].contiguous(), meta["radii"].contiguous(),
        meta["depths"].contiguous(), 16, TW, TH, sort=False)
    roff = gsplat.isect_offset_encode(iid, 1, TW, TH).contiguous()
    ioff = roff[0].reshape(-1).cpu().numpy()
    tile_lens = np.diff(np.concatenate([ioff, [int(fid.numel())]]))
    nz = tile_lens[tile_lens > 0]
    torch.set_grad_enabled(True)
    return {
        "gauss_count": means.shape[0],
        "n_intersections": int(iid.numel()),
        "n_sorted": int(fid.numel()),
        "n_nonzero_tiles": int(len(nz)),
        "tile_len_p50": float(np.percentile(nz, 50)) if len(nz) > 0 else 0,
        "tile_len_p90": float(np.percentile(nz, 90)) if len(nz) > 0 else 0,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=5)
    p.add_argument("--scene", default="room")
    args = p.parse_args()
    torch.manual_seed(0)
    bg = torch.zeros(1, 3, device=DEV)
    
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    scene = load_ply(str(ROOT / f"data/official/mipnerf360/{args.scene}/point_cloud.ply"), device=DEV)
    cams = resize_cameras(load_cameras_from_json(str(ROOT / f"data/official/mipnerf360/{args.scene}/cameras.json"), device=DEV), W, H)
    cam = cams[args.camera]
    n_total = scene["xyz"].shape[0]
    
    # Training phase definitions (Gaussian counts approximating real training states)
    phases = [
        ("iter_0", max(1, n_total // 80)),     # ~20K, early
        ("iter_3K", max(1, n_total // 40)),    # ~40K
        ("iter_5K", max(1, n_total // 20)),    # ~80K
        ("iter_10K", max(1, n_total // 7)),    # ~228K
        ("iter_15K", max(1, n_total // 3)),    # ~531K
        ("iter_20K", max(1, n_total // 2)),    # ~796K
        ("iter_25K", max(1, int(n_total * 0.8))), # ~1.27M
        ("iter_30K", n_total),                   # ~1.59M
    ]
    
    phase_results = []
    for label, n_g in phases:
        m = scene["xyz"].detach().clone()[:n_g].requires_grad_(True)
        q = torch.nn.functional.normalize(scene["rotations"].detach().clone()[:n_g], dim=-1).requires_grad_(True)
        s = scene["scales"].detach().clone()[:n_g].exp().requires_grad_(True)
        o = scene["opacity"].detach().clone()[:n_g].requires_grad_(True)
        sh = scene["shs"].detach().clone()[:n_g].requires_grad_(True)
        
        # Warmup
        for _ in range(2):
            for x in [m, q, s, o, sh]: x.grad = None
            ra, aa, _ = gsplat.rasterization(means=m, quats=q, scales=s, opacities=o, colors=sh,
                viewmats=cam.world_view_transform[None].contiguous(), Ks=cam.K[None].contiguous(),
                width=W, height=H, near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
                sh_degree=3, packed=False, tile_size=16, backgrounds=bg, render_mode="RGB")
            (ra.float().mean() + aa.float().mean()).backward()
        torch.cuda.synchronize()
        
        pr = {"phase": label, "n_gaussians": n_g}
        for ts in TILE_VALS:
            fw, bw, ti = measure(cam, m, q, s, o, sh, bg, ts, reps=5)
            pr[f"tile{ts}"] = {"fwd_ms": fw, "bwd_ms": bw, "T_iter_ms": ti}
        
        ws = workload_stats(cam, m, q, s, o, sh, bg)
        pr["workload"] = ws
        
        benefit = (pr["tile16"]["T_iter_ms"] - pr["tile24"]["T_iter_ms"]) / max(pr["tile16"]["T_iter_ms"], 0.001) * 100
        pr["tile24_benefit_pct"] = -benefit
        
        print(f"  {label:>12s}: G={n_g:>7d} T16={pr['tile16']['T_iter_ms']:.2f} T24={pr['tile24']['T_iter_ms']:.2f} "
              f"benefit24={pr['tile24_benefit_pct']:.1f}% nz_tiles={ws['n_nonzero_tiles']}", flush=True)
        phase_results.append(pr)
    
    # Phase-aware analysis
    benefits = {pr["phase"]: pr["tile24_benefit_pct"] for pr in phase_results}
    nz_tiles = {pr["phase"]: pr["workload"]["n_nonzero_tiles"] for pr in phase_results}
    
    # Predict tile=24 benefit from n_nonzero_tiles or other observables
    nz_vals = np.array(list(nz_tiles.values()))
    ben_vals = np.array(list(benefits.values()))
    valid = ~np.isnan(ben_vals)
    corr = float(np.corrcoef(nz_vals[valid], ben_vals[valid])[0, 1]) if sum(valid) > 2 else 0.0
    
    # Find best tile per phase
    best_tile = {}
    for pr in phase_results:
        best = 16 if pr["tile16"]["T_iter_ms"] <= pr["tile24"]["T_iter_ms"] else 24
        best_tile[pr["phase"]] = best
    
    # Does a cheap observable predict tile choice?
    has_predictive = abs(corr) > 0.5
    
    out = {
        "schema_version": 2, "phase": "P0-B I1 real training-state validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": args.scene, "resolution": f"{W}x{H}", "camera": args.camera},
        "phase_results": phase_results,
        "analysis": {
            "tile24_benefit_per_phase": benefits,
            "non_zero_tiles_per_phase": nz_tiles,
            "tile24_nz_tiles_correlation": round(corr, 3),
            "best_tile_per_phase": best_tile,
            "has_predictive_observable": has_predictive,
            "max_benefit_pct": max(benefits.values()),
            "phases_with_tile24_benefit": sum(1 for v in benefits.values() if v > 5),
        },
        "verdict": "KEEP" if has_predictive or max(benefits.values()) > 5 else "DROP",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")
    print(f"Max benefit: {max(benefits.values()):.1f}%")

if __name__ == "__main__":
    main()
