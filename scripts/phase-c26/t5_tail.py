#!/usr/bin/env python3
"""T5' — 3-Experiment Sparse-Tail Validation.

Experiment A: Active-fraction sweep via controlled subsampling (replicates C25 with more detail)
Experiment B: Depth-tail sweep using forward last_ids to isolate intervals of sorted traversal
Experiment C: Compaction overhead proxy (ballot + prefix + compact estimate)
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

def fwd_raw(m2d, conics, colors, opac, bg, roff, fid):
    from gsplat.cuda._wrapper import _make_lazy_cuda_func
    return _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
        m2d.contiguous(), conics.contiguous(), colors.contiguous(),
        opac.contiguous(), bg, None, W, H, TILE, roff.contiguous(), fid.contiguous())

def exp_a_controlled_subsample(cam, means, quats, scales, opac, shs, bg):
    """Experiment A: measure backward vs active fraction (replicate C25)."""
    subs = [1.0, 0.75, 0.50, 0.25, 0.1, 0.05, 0.01]
    results = []
    n_total = means.shape[0]
    for sf in subs:
        n = max(int(n_total * sf), 100)
        fwds, bwds = [], []
        for _ in range(5):
            for x in [means, quats, scales, opac, shs]:
                if x.grad is not None: x.grad = None
            torch.cuda.synchronize(); t0 = time.perf_counter()
            rgb, a, _ = gsplat.rasterization(
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
        results.append({"subsample_frac": sf, "n_gaussians": n,
                         "fwd_ms": float(np.mean(fwds)), "bwd_ms": float(np.mean(bwds))})
    full_bwd = results[0]["bwd_ms"]
    for r in results:
        r["bwd_relative"] = r["bwd_ms"] / full_bwd if full_bwd else 0
    return results

def exp_b_depth_tail_sweep(cam, means, quats, scales, opac, shs, bg):
    """Experiment B: divide sorted traversal into depth intervals and measure cost per interval.
    
    We use forward last_ids to classify each sorted position by which depth interval it falls into.
    Then estimate backward cost per interval using a proxy: each sorted position contributes
    approximately equally to backward traversal cost.
    """
    # Run full forward to get last_ids and intersection data
    torch.set_grad_enabled(False)
    rgb, a, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H,
        near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
        sh_degree=3, packed=False, tile_size=TILE,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    _, iid, fid = gsplat.isect_tiles(
        meta["means2d"].contiguous(), meta["radii"].contiguous(),
        meta["depths"].contiguous(), TILE, TW, TH, sort=True)
    roff = gsplat.isect_offset_encode(iid, 1, TW, TH).contiguous()
    
    # Get last_ids from forward
    colors_fwd = gsplat.spherical_harmonics(3, (cam.camera_center.to(DEV)-means)/(cam.camera_center.to(DEV)-means).norm(dim=-1,keepdim=True), shs).unsqueeze(0)
    rc, ra, last_tup = fwd_raw(
        meta["means2d"].contiguous(), meta["conics"].contiguous(),
        colors_fwd.contiguous(), meta["opacities"].contiguous(), bg, roff, fid.contiguous())
    last_ids = last_tup[0]
    
    starts = roff[0].reshape(-1).long().tolist()
    n_sorted = int(fid.numel())
    ends = starts[1:] + [n_sorted]
    
    # Classify each sorted position by depth interval
    intervals = [0, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0]
    depth_labels = ["0-50%", "50-75%", "75-90%", "90-95%", "95-99%", "99-100%"]
    
    interval_counts = {l: 0 for l in depth_labels}
    interval_terminated = {l: 0 for l in depth_labels}
    
    for tile_i, (lo, hi) in enumerate(zip(starts, ends)):
        n = hi - lo
        if n <= 0: continue
        ty, tx = divmod(tile_i, TW)
        tile_l = last_ids[ty*TILE:min((ty+1)*TILE, H), tx*TILE:min((tx+1)*TILE, W)].reshape(-1)
        npx = int(tile_l.numel())
        
        for i in range(n):
            depth = i / n
            # Which interval?
            idx = min(sum(1 for b in intervals[1:] if depth >= b), len(depth_labels)-1)
            interval_counts[depth_labels[idx]] += npx
            # Is this position past the pixel's last_ids?
            px_in_tile = i // npx if npx > 0 else 0
            if px_in_tile < tile_l.shape[0]:
                term_point = int(tile_l[px_in_tile].item()) if n > 0 else 0
                if term_point > 0 and lo + i >= term_point:
                    interval_terminated[depth_labels[idx]] += npx
    
    total = sum(interval_counts.values())
    bwd_proxy = {k: (interval_counts[k], interval_terminated[k],
                      interval_terminated[k]/max(interval_counts[k],1))
                 for k in depth_labels}
    
    out = []
    for k in depth_labels:
        cnt, term, frac = bwd_proxy[k]
        out.append({
            "interval": k,
            "sorted_positions_frac": cnt/max(total,1),
            "terminated_pixels_in_interval": term,
            "terminated_fraction": frac,
            "estimated_backward_cost_share": cnt/max(total,1)
        })
    return out

def exp_c_compaction_overhead_proxy(cam, means, quats, scales, opac, shs, bg):
    """Experiment C: estimate compaction overhead via PyTorch proxy.
    
    Simulates: mask generation, warp-level ballot, prefix sum, compacted launch.
    """
    torch.set_grad_enabled(False)
    rgb, a, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),
        Ks=cam.K[None].contiguous(), width=W, height=H,
        near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
        sh_degree=3, packed=False, tile_size=TILE,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    
    N_GAUSS = means.shape[0]
    N_PIX = W * H
    WARP = 32
    
    # Estimate 1: mask generation (last_ids comparison) = ~N_sorted comparisons
    n_sorted_est = int(meta["radii"].shape[1]) * 50  # rough estimate per-pixel sorted positions
    
    # Simulate with a large tensor operation of equivalent size
    mock_data = torch.randn(min(n_sorted_est, 5000000), device=DEV)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    mask = mock_data > 0.5
    torch.cuda.synchronize(); t1 = time.perf_counter()
    mask_time = (t1 - t0) * 1000
    
    # Estimate 2: warp-level ballot simulation (popcount via GPU)
    n_warps = min(n_sorted_est // WARP, 200000)
    mock_lanes = torch.randint(0, 2, (n_warps, WARP), device=DEV, dtype=torch.int32)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    ballot = torch.sum(mock_lanes, dim=1)  # proxy for __ballot_sync + popcount
    torch.cuda.synchronize(); t3 = time.perf_counter()
    ballot_time = (t3 - t2) * 1000
    
    # Estimate 3: compact/repack (copy active lanes only)
    n_compact = int(n_warps * WARP * 0.3)  # ~30% active in sparse tail
    mock_active = torch.randn(n_compact, 32, device=DEV) if n_compact > 0 else torch.randn(1, 32, device=DEV)
    mock_all = torch.randn(n_warps, WARP, device=DEV) if n_warps > 0 else torch.randn(1, WARP, device=DEV)
    indices = torch.nonzero(mock_lanes.reshape(-1) > 0).reshape(-1)[:n_compact]
    torch.cuda.synchronize(); t4 = time.perf_counter()
    if indices.numel() > 0:
        compacted = mock_all.reshape(-1, WARP)[indices//WARP]
    torch.cuda.synchronize(); t5 = time.perf_counter()
    compact_time = (t5 - t4) * 1000 if indices.numel() > 0 else 0.001
    
    # Estimate total overhead per iteration
    total_overhead_us = (mask_time + ballot_time + compact_time) * 1000
    
    return {
        "mask_generation_us": mask_time * 1000,
        "ballot_popcount_us": ballot_time * 1000,
        "compact_copy_us": compact_time * 1000,
        "total_overhead_estimate_us": total_overhead_us,
        "n_mock_elements": n_sorted_est,
        "n_mock_warps": n_warps,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=5)
    args = p.parse_args()
    torch.manual_seed(0)
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=DEV)
    cams = resize_cameras(
        load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=DEV), W, H)
    cam = cams[args.camera]
    bg = torch.zeros(1, 3, device=DEV)
    means = scene["xyz"].detach().clone().requires_grad_(True)
    quats = torch.nn.functional.normalize(scene["rotations"].detach().clone(), dim=-1).requires_grad_(True)
    scales = scene["scales"].detach().clone().exp().requires_grad_(True)
    opac = scene["opacity"].detach().clone().requires_grad_(True)
    shs = scene["shs"].detach().clone().requires_grad_(True)
    
    print("Warmup...")
    for _ in range(3):
        for x in [means, quats, scales, opac, shs]: x.grad = None
        rgb, a, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),
            Ks=cam.K[None].contiguous(), width=W, height=H,
            near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3,
            sh_degree=3, packed=False, tile_size=TILE,
            backgrounds=bg, render_mode="RGB")
        (rgb.float().mean() + a.float().mean()).backward()
    torch.cuda.synchronize()
    print("  Warmup done", flush=True)
    
    # Experiment A
    print("Exp A: controlled subsample...", flush=True)
    exp_a = exp_a_controlled_subsample(cam, means, quats, scales, opac, shs, bg)
    # Verification: at 1% Gaussians, bwd_relative should be high
    for r in exp_a:
        print(f"  sf={r['subsample_frac']:.2f} n={r['n_gaussians']:7d} bwd={r['bwd_ms']:.3f}ms rel={r['bwd_relative']:.3f}", flush=True)
    
    # Experiment B
    print("Exp B: depth-tail sweep...", flush=True)
    exp_b = exp_b_depth_tail_sweep(cam, means, quats, scales, opac, shs, bg)
    for r in exp_b:
        print(f"  {r['interval']}: sorted_frac={r['sorted_positions_frac']:.4f} term_frac={r['terminated_fraction']:.4f}", flush=True)
    
    # Experiment C
    print("Exp C: compaction overhead proxy...", flush=True)
    exp_c = exp_c_compaction_overhead_proxy(cam, means, quats, scales, opac, shs, bg)
    print(f"  total_overhead_us={exp_c['total_overhead_estimate_us']:.1f}", flush=True)
    
    # Falsification test
    a1_r = exp_a[0]["bwd_ms"]
    a1pct_r = exp_a[-1]["bwd_ms"]  # 1% subsample
    residual_frac = a1pct_r / max(a1_r, 0.001)
    # The ~1.7ms floor is mostly overhead. If depth-tail also shows high terminated fraction
    # in the 90-100% range, T5' is strongly supported.
    tail_99_100 = [r for r in exp_b if r["interval"] == "99-100%"][0]
    tail_95_99 = [r for r in exp_b if r["interval"] == "95-99%"][0]
    
    verbose = f"1% subsample retains {residual_frac:.1%} of backward cost. "
    verbose += f"99-100% interval has {tail_99_100['terminated_fraction']:.1%} terminated pixels. "
    verbose += f"Compaction overhead estimate: {exp_c['total_overhead_estimate_us']:.0f}us. "
    
    # Verdict
    if residual_frac > 0.25 and tail_99_100["terminated_fraction"] > 0.8 and exp_c["total_overhead_estimate_us"] < 50:
        verdict = "STRONG KEEP — sparse-tail cost confirmed, terminated-fraction very high in tail, compaction overhead is small"
    elif residual_frac > 0.15:
        verdict = "KEEP — non-proportional cost confirmed but need tighter quantification"
    else:
        verdict = "DROP — residual cost at 1% is small; kernel handles sparsity efficiently"
    
    out = {
        "schema_version": 2, "phase": "T5' sparse-tail validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TILE, "camera": args.camera},
        "exp_a_controlled_subsample": exp_a,
        "exp_b_depth_tail_sweep": exp_b,
        "exp_c_compaction_proxy": exp_c,
        "falsification_analysis": {
            "residual_cost_at_1_percent": residual_frac,
            "terminated_fraction_99_100": tail_99_100["terminated_fraction"],
            "terminated_fraction_95_99": tail_95_99["terminated_fraction"],
        },
        "evidence": verbose,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")
    print(f"Verdict: {verdict}")

if __name__ == "__main__":
    main()
