#!/usr/bin/env python3
"""
Track A: Causal Host CUDA Query Analysis — Minimal Microbenchmark.

Measures the actual wall-clock impact of repeated CUDA metadata/occupancy queries.
Strategy: Compare T_iter of "cold" iteration (no kernel warmup → many first-time
metadata queries) vs "hot" iteration (all occupancy/attribute results cached inside
CUDA driver after warmup).

Pre-registration (from existing trace analysis):
  - cudaDeviceGetAttribute:  186 calls/iter, 0.057ms CPU (0.3us each)
  - cudaFuncGetAttributes:    45 calls/iter, 0.096ms CPU (2.1us each)
  - cudaOccupancyMaxActive...: 51 calls/iter, 0.040ms CPU (0.8us each)
  - cudaFuncSetAttribute:     17 calls/iter, 0.023ms CPU (1.3us each)
  - Total metadata queries:  ~368 calls/iter, ~0.22ms CPU
  - Total non-sync CUDA API: ~350 launch + 15 memcpy + 55 memset + 368 queries
                              ≈ 5.7ms/iter CPU time

  H0: Caching these queries saves <0.5% of T_iter
  H1: Caching these queries saves >5% of T_iter

Method:
  - Cold block: No warmup, 1 iter per kernel recompilation (each iter uses
    slightly different tensor sizes due to caching allocator)
  - Hot block: 50+ warmup iters, then 100 measured iters (driver caches fully warm)
  - Compare T_iter cold vs hot

Prediction: Cold ≈ Hot (within noise), because queries are per-unique-config,
not repeated for the same config.
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

TILE_SIZE = 16

def make_opt(model, sls):
    return torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * sls},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ], eps=1e-15, betas=(0.9, 0.999))

def create(device="cuda:0", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dataset = GTDataset(scene="room", repo_root=ROOT, resolution="1080p", device=device)
    sfm = load_initial_checkpoint("room", ROOT, device=device)
    n_cam = len(dataset)
    model = GaussianModel(num_points=sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((sfm["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm.get("scales"), rotations_raw=sfm.get("rotations"), shs=sfm.get("shs"))
    sls = float(sfm["xyz"].norm(dim=-1).max().item())
    opt = make_opt(model, sls)
    return dataset, model, opt, n_cam, sls

def run_block(dataset, model, opt, n_cam, sls, n_iters, device="cuda:0", warmup=False):
    """Run a block of iterations with block timing. No metrics, no sync."""
    for step in range(n_iters):
        ci = step % n_cam
        cam = dataset.get_camera(ci); gt = dataset.get_gt_image(ci)

        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=TILE_SIZE, packed=True, sh_degree=0,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            sparse_grad=False, absgrad=False,
        )
        rendered = rendered[0].clamp(0, 1)
        ld = combined_loss(rendered, gt, lambda_dssim=0.2)
        loss = ld["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # Densification (same as Phase-7)
        if step >= 200 and step < 30000 and step % 100 == 0:
            denf = model.densification(grad_threshold=2e-4, clone_max_screen_size=20, split_max_screen_size=20)
            pruned = model.prune_and_reset(opacity_threshold=0.005, reset_interval=3000, current_step=step)
            tot = denf["cloned"] + denf["split"] + pruned
            if tot > 0:
                new_opt = make_opt(model, sls)
                new_opt.load_state_dict(opt.state_dict())
                opt = new_opt
                for pg in opt.param_groups:
                    pg["params"] = [model.xyz, model.rotations, model.scales, model.opacity, model.shs]

    return opt  # return updated opt in case densification occurred

def count_query_stats():
    """Use CUDA driver introspection to count metadata queries (rough estimate).
    Uses torch.cuda.caching_allocator stats as proxy for allocator activity.
    """
    stats = {}
    for k, v in torch.cuda.memory_stats().items():
        if "segment" in k or "active" in k or "alloc" in k or "request" in k or "free" in k:
            stats[k] = v
    return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/phase-c31/c33_a_query_overhead.json")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    print(f"=== C33-A: Host CUDA Query Overhead ===")
    print(f"Device: {torch.cuda.get_device_properties(args.gpu).name}")
    print()

    results = {}

    # ── Cold block (no warmup) ──
    print("--- Cold Block (no warmup) ---")
    dataset, model, opt, n_cam, sls = create(device=device, seed=42)
    mem_before = torch.cuda.memory_allocated() / 1024**2

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    opt = run_block(dataset, model, opt, n_cam, sls, n_iters=10, device=device)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000 / 10
    mem_after = torch.cuda.memory_allocated() / 1024**2
    print(f"  T_iter (cold): {cold_ms:.3f}ms  (mem: {mem_before:.0f}→{mem_after:.0f}MB)")
    results["cold"] = {"t_iter_ms": round(cold_ms, 3), "mem_before_mb": round(mem_before, 1), "mem_after_mb": round(mem_after, 1)}

    # ── Warmup → Hot block ──
    print("--- Hot Block (50 warmup + 100 measured) ---")
    dataset, model, opt, n_cam, sls = create(device=device, seed=42)
    # Warmup
    opt = run_block(dataset, model, opt, n_cam, sls, n_iters=50, device=device)
    gc.collect()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1024**2
    # Cuda driver caches should be warm now

    # Record memory stats before measurement
    stats_before = count_query_stats()

    # Measured block
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    opt = run_block(dataset, model, opt, n_cam, sls, n_iters=100, device=device)
    torch.cuda.synchronize()
    hot_ms = (time.perf_counter() - t0) * 1000 / 100

    stats_after = count_query_stats()
    mem_after = torch.cuda.memory_allocated() / 1024**2

    # Memory stats delta
    stats_delta = {}
    for k in stats_before:
        if k in stats_after:
            stats_delta[k] = stats_after[k] - stats_before[k]

    print(f"  T_iter (hot):  {hot_ms:.3f}ms  (mem: {mem_before:.0f}→{mem_after:.0f}MB)")
    print(f"  Memory stats (delta over 100 iters):")
    for k, v in sorted(stats_delta.items()):
        print(f"    {k}: {v}")
    results["hot"] = {"t_iter_ms": round(hot_ms, 3), "mem_before_mb": round(mem_before, 1), "mem_after_mb": round(mem_after, 1)}
    results["mem_stats_delta"] = {k: v for k, v in stats_delta.items() if v != 0}
    results["cold_vs_hot_diff_ms"] = round(cold_ms - hot_ms, 3)
    results["cold_vs_hot_pct"] = round((cold_ms - hot_ms) / hot_ms * 100, 3)

    print()
    print(f"=== Conclusion ===")
    print(f"  Cold T_iter: {cold_ms:.3f}ms")
    print(f"  Hot  T_iter: {hot_ms:.3f}ms")
    print(f"  Diff: {cold_ms - hot_ms:.3f}ms ({results['cold_vs_hot_pct']:.3f}%)")
    if results["cold_vs_hot_pct"] < 1.0:
        print("  → DROP: Cold ≈ Hot. Metadata queries are NOT a significant overhead.")
        print("    (Trace analysis predicted <0.25% from metadata queries. Confirmed.)")
    else:
        print(f"  → {results['cold_vs_hot_pct']:.1f}% difference — further analysis needed.")
    print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
