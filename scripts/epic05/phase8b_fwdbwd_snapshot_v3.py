#!/usr/bin/env python3
"""
Phase 8B — Real-scene snapshot forward+backward microbenchmark (v3).

Key methodology changes from v2:
  - Median-based primary timing (robust to GPU throttling outliers)
  - Short BATCH for tile16 fwd+bwd (2-4 samples per repeat, 2 repeats)
  - Run each checkpoint individually (no combined script timeout)
  - Report ALL individual samples for diagnosis of variance
  - Tile32 fwd+bwd still uses BATCH=5 for good statistics
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gsplat import rasterization

DEVICE = "cuda"
DTYPE = torch.float32


def make_camera(W=1920, H=1080):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    viewmat = torch.eye(4, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    t = torch.eye(4, device=DEVICE, dtype=DTYPE)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor(
        [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
        device=DEVICE, dtype=DTYPE,
    ).unsqueeze(0)
    return viewmat, K, W, H


def make_multiple_cameras(W=1920, H=1080):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    cams = []
    for z_offset, x_angle_deg in [(-5.0, 0), (-4.5, -10), (-6.0, 15)]:
        viewmat = torch.eye(4, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        t = torch.eye(4, device=DEVICE, dtype=DTYPE)
        t[0, 3] = math.tan(math.radians(x_angle_deg)) * abs(z_offset)
        t[2, 3] = z_offset
        viewmat[0] = t
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
            device=DEVICE, dtype=DTYPE,
        ).unsqueeze(0)
        cams.append((viewmat, K, W, H))
    return cams


def load_checkpoint(path, device=DEVICE):
    cp = torch.load(path, map_location=device, weights_only=False)
    ms = cp["model_state"]
    opac = ms["opacity"].detach().clone()
    if opac.dim() == 2 and opac.shape[1] == 1:
        opac = opac.squeeze(1)
    return {
        "xyz": ms["xyz"].detach().clone(),
        "rotations": ms["rotations"].detach().clone(),
        "scales": ms["scales"].detach().clone(),
        "opacity": opac,
        "shs": ms["shs"].detach().clone(),
        "num_points": ms["num_points"],
        "sh_degree": ms["sh_degree"],
    }


def compute_workload_stats(params, viewmat, K, W, H, tile_size):
    with torch.no_grad():
        rendered, alpha, meta = rasterization(
            means=params["xyz"], quats=params["rotations"],
            scales=params["scales"], opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
        )
    tpg = meta["tiles_per_gauss"]
    tpg_np = tpg.float().cpu().numpy()
    tile_w = meta["tile_width"]
    tile_h = meta["tile_height"]
    total_tiles = tile_w * tile_h
    return {
        "total_gaussians": params["xyz"].shape[0],
        "nnz_gaussians": tpg.shape[0],
        "tile_grid_width": tile_w,
        "tile_grid_height": tile_h,
        "total_tiles": total_tiles,
        "tpg_mean": float(tpg.float().mean().item()),
        "tpg_median": float(tpg.float().median().item()),
        "tpg_std": float(tpg.float().std().item()),
        "tpg_min": int(tpg.min().item()),
        "tpg_max": int(tpg.max().item()),
        "tpg_p95": float(np.percentile(tpg_np, 95)),
        "tpg_p99": float(np.percentile(tpg_np, 99)),
        "total_intersections": int(tpg.sum().item()),
        "gaussians_per_tile_mean": float(tpg.sum().item() / total_tiles),
    }, meta


def timing_forward(params, viewmat, K, W, H, tile_size, n_batch=10, n_repeat=3, warmup=3):
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(warmup):
        rendered, alpha, meta = rasterization(
            means=params["xyz"], quats=params["rotations"],
            scales=params["scales"], opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
        )
        torch.cuda.synchronize()
    for r in range(n_repeat):
        for b in range(n_batch):
            start_ev.record()
            rendered, alpha, meta = rasterization(
                means=params["xyz"], quats=params["rotations"],
                scales=params["scales"], opacities=params["opacity"],
                colors=params["shs"],
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
            )
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    return np.array(times), meta


def timing_fwd_bwd(params, viewmat, K, W, H, tile_size, n_batch=5, n_repeat=2, warmup=2):
    xyz = params["xyz"].detach().clone().requires_grad_(True)
    rotations = params["rotations"].detach().clone().requires_grad_(True)
    scales = params["scales"].detach().clone().requires_grad_(True)
    opacity = params["opacity"].detach().clone().requires_grad_(True)
    shs = params["shs"].detach().clone().requires_grad_(True)
    target = torch.rand(1, H, W, 3, device=DEVICE, dtype=DTYPE)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        rendered, alpha, meta = rasterization(
            means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
        )
        loss = ((rendered - target) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        for p in [xyz, rotations, scales, opacity, shs]:
            if p.grad is not None:
                p.grad = None

    times = []
    for r in range(n_repeat):
        for b in range(n_batch):
            start_ev.record()
            rendered, alpha, meta = rasterization(
                means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
            )
            loss = ((rendered - target) ** 2).mean()
            loss.backward()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
            for p in [xyz, rotations, scales, opacity, shs]:
                if p.grad is not None:
                    p.grad = None

    # Grad check (one extra iter)
    for p in [xyz, rotations, scales, opacity, shs]:
        if p.grad is not None:
            p.grad = None
    rendered, alpha, _ = rasterization(
        means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
        viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
    )
    loss = ((rendered - target) ** 2).mean()
    loss.backward()
    torch.cuda.synchronize()

    grad_check = {
        "xyz_requires_grad": xyz.requires_grad,
        "xyz_grad_finite": bool(torch.isfinite(xyz.grad).all().item()),
        "xyz_grad_any_nonzero": bool((xyz.grad.abs() > 0).any().item()),
        "xyz_grad_norm": float(xyz.grad.norm().item()),
        "xyz_grad_shape": list(xyz.grad.shape),
        "loss_value": float(loss.item()),
    }
    for name, p in [("rotations", rotations), ("scales", scales),
                     ("opacity", opacity), ("shs", shs)]:
        if p.grad is None:
            grad_check[f"{name}_has_grad"] = False
        else:
            grad_check[f"{name}_has_grad"] = True
            grad_check[f"{name}_grad_finite"] = bool(torch.isfinite(p.grad).all().item())
            grad_check[f"{name}_grad_norm"] = float(p.grad.norm().item())

    return np.array(times), grad_check


def compute_robust_stats(arr):
    """Median-based primary metrics with all raw samples reported."""
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
        "n_samples": int(len(arr)),
        "raw_samples_ms": [float(x) for x in arr],
    }


def run_tile_size(params, viewmat, K, W, H, tile_size, is_tile16=False):
    print(f"  --- tile_size={tile_size} ---", flush=True)

    wl_stats, meta = compute_workload_stats(params, viewmat, K, W, H, tile_size)
    print(f"    Workload: {wl_stats['total_intersections']/1e6:.1f}M intersections, "
          f"tpg={wl_stats['tpg_mean']:.1f} (max={wl_stats['tpg_max']})", flush=True)

    # Forward timing — generous sampling
    print(f"    Forward timing...", flush=True)
    fwd_times, _ = timing_forward(params, viewmat, K, W, H, tile_size,
                                   n_batch=10, n_repeat=3, warmup=3)
    fwd_stats = compute_robust_stats(fwd_times)
    print(f"    Forward: {fwd_stats['median_ms']:.2f} ms (median)   "
          f"{fwd_stats['mean_ms']:.2f} ± {fwd_stats['std_ms']:.2f} ms (mean±std)   "
          f"[{fwd_stats['min_ms']:.2f}, {fwd_stats['max_ms']:.2f}]  CV={fwd_stats['cv']:.4f}", flush=True)

    # Fwd+Bwd timing — fewer samples for tile16 (very slow)
    print(f"    Fwd+Bwd timing...", flush=True)
    if is_tile16:
        fb_n_batch = 3
        fb_n_repeat = 2
    else:
        fb_n_batch = 5
        fb_n_repeat = 2
    fb_warmup = 2

    fwd_bwd_times, grad_check = timing_fwd_bwd(params, viewmat, K, W, H, tile_size,
                                                 n_batch=fb_n_batch, n_repeat=fb_n_repeat,
                                                 warmup=fb_warmup)
    fb_stats = compute_robust_stats(fwd_bwd_times)
    print(f"    Fwd+Bwd: {fb_stats['median_ms']:.2f} ms (median)   "
          f"{fb_stats['mean_ms']:.2f} ± {fb_stats['std_ms']:.2f} ms (mean±std)   "
          f"[{fb_stats['min_ms']:.2f}, {fb_stats['max_ms']:.2f}]  CV={fb_stats['cv']:.4f}", flush=True)

    # Infer backward using median (more robust)
    bwd_median = fb_stats["median_ms"] - fwd_stats["median_ms"]
    bwd_mean = fb_stats["mean_ms"] - fwd_stats["mean_ms"]
    bwd_var = max(0.001, fb_stats["std_ms"]**2 - fwd_stats["std_ms"]**2)
    bwd_stats = {
        "median_ms": max(0.001, bwd_median),
        "mean_ms": max(0.001, bwd_mean),
        "std_ms": math.sqrt(bwd_var),
        "cv": math.sqrt(bwd_var) / max(0.001, bwd_mean),
    }
    print(f"    Inferred Bwd (median): {bwd_stats['median_ms']:.2f} ms  "
          f"(mean: {bwd_stats['mean_ms']:.2f} ms)", flush=True)
    print(f"    Grad check: all finite={grad_check['xyz_grad_finite']}, "
          f"xyz_norm={grad_check['xyz_grad_norm']:.6f}", flush=True)

    return {
        "forward": fwd_stats,
        "inferred_backward": bwd_stats,
        "forward_plus_backward": fb_stats,
        "gradient_verification": grad_check,
        "workload_statistics": wl_stats,
        "tile_grid": f"{wl_stats['tile_grid_width']}x{wl_stats['tile_grid_height']}",
    }


def run_single_checkpoint(params, viewmat, K, W, H, ckpt_name):
    entry = {
        "num_gaussians": params["xyz"].shape[0],
        "sh_degree": params["sh_degree"],
        "tile_sizes": {},
    }

    tile16 = run_tile_size(params, viewmat, K, W, H, 16, is_tile16=True)
    entry["tile_sizes"]["16"] = tile16
    torch.cuda.empty_cache()

    tile32 = run_tile_size(params, viewmat, K, W, H, 32, is_tile16=False)
    entry["tile_sizes"]["32"] = tile32
    torch.cuda.empty_cache()

    # Ratios using MEDIAN (robust to outliers)
    ratios = {}
    for phase, key in [("forward", "forward"), ("forward_plus_backward", "forward_plus_backward")]:
        r = tile16[key]["median_ms"] / tile32[key]["median_ms"]
        ratios[phase] = {
            "ratio_t16_t32_median": r,
            "ratio_t16_t32_mean": tile16[key]["mean_ms"] / tile32[key]["mean_ms"],
            "speedup_t32": 1.0 / r,
            "t16_median_ms": tile16[key]["median_ms"],
            "t32_median_ms": tile32[key]["median_ms"],
            "t16_mean_ms": tile16[key]["mean_ms"],
            "t32_mean_ms": tile32[key]["mean_ms"],
        }

    br_mdn = tile16["inferred_backward"]["median_ms"] / tile32["inferred_backward"]["median_ms"]
    ratios["inferred_backward"] = {
        "ratio_t16_t32_median": br_mdn,
        "ratio_t16_t32_mean": tile16["inferred_backward"]["mean_ms"] / tile32["inferred_backward"]["mean_ms"],
        "t16_median_ms": tile16["inferred_backward"]["median_ms"],
        "t32_median_ms": tile32["inferred_backward"]["median_ms"],
        "t16_mean_ms": tile16["inferred_backward"]["mean_ms"],
        "t32_mean_ms": tile32["inferred_backward"]["mean_ms"],
    }

    entry["ratios"] = ratios

    br_str = f"{br_mdn:.2f}"
    print(f"\n  >>> RATIOS (median): Forward={ratios['forward']['ratio_t16_t32_median']:.2f}x  "
          f"Backward={br_str}x  "
          f"Fwd+Bwd={ratios['forward_plus_backward']['ratio_t16_t32_median']:.2f}x", flush=True)

    return entry


def run_single_checkpoint_only():
    """Run a single checkpoint (called via CLI arg)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_name", help="e.g. room_iter5000, room_iter10000, ...")
    parser.add_argument("--multi-camera", action="store_true", help="Also run 3 cameras")
    args = parser.parse_args()

    ckpt_dir = REPO_ROOT / "results" / "epic05" / "phase7"
    output_dir = REPO_ROOT / "results" / "epic05" / "phase8b"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_map = {
        "room_iter5000":  "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt",
        "room_iter10000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter10000.pt",
        "room_iter15000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter15000.pt",
        "room_iter20000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter20000.pt",
        "room_iter25000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter25000.pt",
        "room_iter30000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter30000.pt",
    }

    if args.checkpoint_name not in ckpt_map:
        print(f"Unknown checkpoint: {args.checkpoint_name}. Options: {list(ckpt_map.keys())}")
        sys.exit(1)

    ckpt_path = str(ckpt_dir / ckpt_map[args.checkpoint_name])
    if not os.path.exists(ckpt_path):
        print(f"Not found: {ckpt_path}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  LOAD: {args.checkpoint_name}")
    print(f"  PATH: {ckpt_path}")
    print(f"{'='*70}")

    params = load_checkpoint(ckpt_path)
    N = params["xyz"].shape[0]
    print(f"  N={N}, SH deg={params['sh_degree']}")

    viewmat, K, W, H = make_camera()
    entry = run_single_checkpoint(params, viewmat, K, W, H, args.checkpoint_name)

    result = {
        "experiment_id": "phase8b-single-checkpoint",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_name": args.checkpoint_name,
        "config": {"resolution": f"{W}x{H}"},
        "result": entry,
    }

    out_path = output_dir / f"single_{args.checkpoint_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    if args.multi_camera:
        print(f"\n\n{'='*70}")
        print(f"  MULTI-CAMERA TEST")
        print(f"{'='*70}")
        multi_cams = make_multiple_cameras()
        mc_results = {}
        for i, (cv, ck, cw, ch) in enumerate(multi_cams):
            cam_label = f"camera_{i}"
            print(f"\n  === {cam_label} ===")
            mc_results[cam_label] = run_single_checkpoint(params, cv, ck, cw, ch,
                                                           f"{args.checkpoint_name}_{cam_label}")
            torch.cuda.empty_cache()
        mc_out_path = output_dir / f"multicam_{args.checkpoint_name}.json"
        with open(mc_out_path, "w") as f:
            json.dump({"checkpoint": args.checkpoint_name, "cameras": mc_results}, f, indent=2, default=str)
        print(f"\nSaved: {mc_out_path}")

    print(f"\nDone. GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    run_single_checkpoint_only()
