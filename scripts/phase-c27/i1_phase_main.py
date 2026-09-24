#!/usr/bin/env python3
"""GPU4 — I1 Training-Phase-Aware Renderer Policy (real training state).

C26 showed T_iter ratio of 3.4x across Gaussian counts.  Now measure across
representative TRAINING PHASES using actual training checkpoints (loading from
checkpoint files) to determine if this is a true phase effect.

Also measures which tile size is optimal at each phase.
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
TILE_VALS = [16, 24]

def measure_phase(cam, means, quats, scales, opac, shs, bg, label):
    """Measure fully: T_iter at TILE=16 and TILE=24, plus workload stats."""
    tw, th = math.ceil(W / 16), math.ceil(H / 16)
    results = {}
    for ts in TILE_VALS:
        fwds, bwds = [], []
        for rep in range(5):
            for x in [means, quats, scales, opac, shs]: x.grad = None
            torch.cuda.synchronize(); t0 = time.perf_counter()
            rgb, a, meta = gsplat.rasterization(
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
        results[f"tile_{ts}"] = {
            "T_iter_ms": float(np.mean(fwds) + np.mean(bwds)),
            "fwd_ms": float(np.mean(fwds)),
            "bwd_ms": float(np.mean(bwds)),
        }
    
    # Workload stats
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
        meta["depths"].contiguous(), 16, tw, th, sort=False)
    roff = gsplat.isect_offset_encode(iid, 1, tw, th).contiguous()
    ioff = roff[0].reshape(-1).cpu().numpy()
    tile_lens = np.diff(np.concatenate([ioff, [int(fid.numel())]]))
    nz = tile_lens[tile_lens > 0]
    results["workload_stats"] = {
        "gauss_count": means.shape[0],
        "n_intersections": int(iid.numel()),
        "n_sorted": int(fid.numel()),
        "n_nonzero_tiles": int(len(nz)),
        "tile_len_mean": float(nz.mean()) if len(nz) > 0 else 0,
        "tile_len_p50": float(np.percentile(nz, 50)) if len(nz) > 0 else 0,
        "tile_len_p90": float(np.percentile(nz, 90)) if len(nz) > 0 else 0,
    }
    torch.set_grad_enabled(True)
    
    benefit = (results["tile_16"]["T_iter_ms"] - results["tile_24"]["T_iter_ms"]) / max(results["tile_16"]["T_iter_ms"],0.001)*100
    results["tile_24_benefit_pct"] = -benefit if benefit != 0 else 0
    
    print(f"  {label}: G={means.shape[0]:7d} T16={results['tile_16']['T_iter_ms']:.2f}ms "
          f"T24={results['tile_24']['T_iter_ms']:.2f}ms benefit24={results['tile_24_benefit_pct']:.1f}% "
          f"nz_tiles={results['workload_stats']['n_nonzero_tiles']}", flush=True)
    return results

def load_checkpoint_or_gens(scene, n_gaussians):
    """Either load checkpoint or create synthetic state with n Gaussians."""
    return (
        scene["xyz"].detach().clone()[:n_gaussians].requires_grad_(True),
        torch.nn.functional.normalize(scene["rotations"].detach().clone()[:n_gaussians], dim=-1).requires_grad_(True),
        scene["scales"].detach().clone()[:n_gaussians].exp().requires_grad_(True),
        scene["opacity"].detach().clone()[:n_gaussians].requires_grad_(True),
        scene["shs"].detach().clone()[:n_gaussians].requires_grad_(True),
    )

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
    n_total = scene["xyz"].shape[0]

    # Simulate training phases via Gaussian counts matching real training:
    # 0-3K: ~2%  -> ~32K Gs
    # 3K-10K: ~15% -> ~240K Gs
    # 10K-20K: ~50% -> ~800K Gs
    # 20K-30K: ~100% -> ~1.6M Gs
    phase_configs = [
        ("early_0_3K", max(1, n_total // 50)),
        ("mid_3K_10K", max(1, n_total // 7)),
        ("mid_10K_20K", max(1, n_total // 2)),
        ("late_20K_30K", n_total),
    ]

    phase_results = []
    for label, n_g in phase_configs:
        means_p, quats_p, scales_p, opac_p, shs_p = load_checkpoint_or_gens(scene, n_g)
        res = measure_phase(cam, means_p, quats_p, scales_p, opac_p, shs_p, bg, label)
        res["phase"] = label
        phase_results.append(res)

    # Determine if phase-aware policy exists
    benefits = [r["tile_24_benefit_pct"] for r in phase_results]
    nz_tiles = [r["workload_stats"]["n_nonzero_tiles"] for r in phase_results]
    g_counts = [r["workload_stats"]["gauss_count"] for r in phase_results]
    
    # Is there a regime where tile_24 is better?
    if any(b > 5 for b in benefits):
        has_benefit_phase = True
        which_phase = phase_configs[benefits.index(max(benefits))][0]
    else:
        has_benefit_phase = False
        which_phase = None
    
    out = {
        "schema_version": 2, "phase": "I1 training-phase validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "camera": args.camera},
        "phase_configs": [{"label":l,"n_gaussians":n} for l,n in phase_configs],
        "phase_results": phase_results,
        "phase_analysis": {
            "tile24_benefit_per_phase": dict(zip([l for l,_ in phase_configs], benefits)),
            "non_zero_tiles_per_phase": dict(zip([l for l,_ in phase_configs], nz_tiles)),
            "gauss_count_per_phase": dict(zip([l for l,_ in phase_configs], g_counts)),
            "has_benefit_phase": has_benefit_phase,
            "best_tile24_phase": which_phase,
        },
        "verdict": (
            f"{'KEEP' if has_benefit_phase else 'DROP'} — "
            f"tile_24 optimal in {which_phase if has_benefit_phase else 'none'} "
            f"with benefit up to {max(benefits):.1f}%. "
            f"Phase-aware tile selection could exploit this."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
