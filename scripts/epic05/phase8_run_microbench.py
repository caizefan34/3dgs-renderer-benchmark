#!/usr/bin/env python3
"""
Phase 8 — Real-scene snapshot microbenchmark using local training checkpoints.

Purpose:
    Run tile16 vs tile32 timing on real Gaussian distributions from actual training
    checkpoints. This is the definitive test for H5 (scene-dependence hypothesis),
    as synthetic random Gaussians failed to reproduce the 1.58× training speedup.

Usage:
    python scripts/epic05/phase8_run_microbench.py

Output:
    results/epic05/phase8/real_scene_microbench.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gsplat import rasterization

DEVICE = "cuda"
BATCH = 5        # 5 forward calls per repeat (real checkpoints are ~180ms/iter at 900K Gs)
N_REPEAT = 3      # 3 × 5 = 15 forward calls per data point
WARMUP = 2        # 2 warmup iterations


def make_camera(W=1920, H=1080, device=DEVICE):
    """Create a single centered camera."""
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    t = torch.eye(4, device=device, dtype=torch.float32)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor(
        [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return viewmat, K, W, H


def load_checkpoint(path: str, device=DEVICE) -> dict:
    """Load a training checkpoint and extract model params on device."""
    cp = torch.load(path, map_location=device, weights_only=False)
    ms = cp["model_state"]
    opac = ms["opacity"]
    # gsplat expects opacities shape [N] not [N, 1]
    if opac.dim() == 2 and opac.shape[1] == 1:
        opac = opac.squeeze(1)
    return {
        "xyz": ms["xyz"],
        "rotations": ms["rotations"],
        "scales": ms["scales"],
        "opacity": opac,
        "shs": ms["shs"],
        "num_points": ms["num_points"],
        "sh_degree": ms["sh_degree"],
    }


def timing_forward(params: dict, viewmat, K, W, H, tile_size: int, packed: bool = True):
    """Time forward pass with CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print(f"      Warmup ({WARMUP})...", flush=True)
    # Warmup
    for _ in range(WARMUP):
        rendered, alpha, meta = rasterization(
            means=params["xyz"],
            quats=params["rotations"],
            scales=params["scales"],
            opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            tile_size=tile_size,
            packed=packed,
            sh_degree=params["sh_degree"],
        )
        torch.cuda.synchronize()

    print(f"      Measuring forward ({N_REPEAT}×{BATCH})...", flush=True)
    # Measured iterations
    times = []
    for r in range(N_REPEAT):
        for b in range(BATCH):
            start.record()
            rendered, alpha, meta = rasterization(
                means=params["xyz"],
                quats=params["rotations"],
                scales=params["scales"],
                opacities=params["opacity"],
                colors=params["shs"],
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                tile_size=tile_size,
                packed=packed,
                sh_degree=params["sh_degree"],
            )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        print(f"        repeat {r+1}/{N_REPEAT} done", flush=True)

    return np.array(times), meta


def timing_fwd_bwd(params: dict, viewmat, K, W, H, tile_size: int, packed: bool = True):
    """Time forward + backward pass."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Create a fresh copy that requires grad for all trainable params
    xyz = params["xyz"].detach().clone().requires_grad_(True)
    rotations = params["rotations"].detach().clone().requires_grad_(True)
    scales = params["scales"].detach().clone().requires_grad_(True)
    opacity = params["opacity"].detach().clone().requires_grad_(True)
    shs = params["shs"].detach().clone().requires_grad_(True)

    # Loss target (random image) — gsplat renders in NHWC format [B, H, W, C]
    target = torch.rand(1, H, W, 3, device=DEVICE)

    print(f"      Warmup fwd+bwd ({WARMUP})...", flush=True)
    for _ in range(WARMUP):
        rendered, alpha, meta = rasterization(
            means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=packed, sh_degree=params["sh_degree"],
        )
        loss = ((rendered - target) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()

    print(f"      Measuring fwd+bwd ({N_REPEAT}×{BATCH})...", flush=True)
    times = []
    for r in range(N_REPEAT):
        for b in range(BATCH):
            start.record()
            rendered, alpha, meta = rasterization(
                means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=tile_size, packed=packed, sh_degree=params["sh_degree"],
            )
            loss = ((rendered - target) ** 2).mean()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            # Zero grads for next iteration
            xyz.grad = None
            rotations.grad = None
            scales.grad = None
            opacity.grad = None
            shs.grad = None
        print(f"        repeat {r+1}/{N_REPEAT} done", flush=True)

    return np.array(times), meta


def compute_workload_stats(params, viewmat, K, W, H, tile_size: int):
    """Compute per-tile workload statistics."""
    with torch.no_grad():
        rendered, alpha, meta = rasterization(
            means=params["xyz"],
            quats=params["rotations"],
            scales=params["scales"],
            opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            tile_size=tile_size,
            packed=True,
            sh_degree=params["sh_degree"],
        )

    tiles_per_gauss = meta["tiles_per_gauss"]  # [nnz]
    tile_width = meta["tile_width"]
    tile_height = meta["tile_height"]

    stats = {
        "total_gaussians": params["xyz"].shape[0],
        "nnz_gaussians": tiles_per_gauss.shape[0],
        "tile_width": tile_width,
        "tile_height": tile_height,
        "total_tiles": tile_width * tile_height,
        "tiles_per_gauss_mean": tiles_per_gauss.float().mean().item(),
        "tiles_per_gauss_median": tiles_per_gauss.float().median().item(),
        "tiles_per_gauss_std": tiles_per_gauss.float().std().item(),
        "tiles_per_gauss_min": tiles_per_gauss.min().item(),
        "tiles_per_gauss_max": tiles_per_gauss.max().item(),
        "total_intersections": tiles_per_gauss.sum().item(),
    }
    return stats


def main():
    output_dir = REPO_ROOT / "results" / "epic05" / "phase8"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints directory
    ckpt_dir = REPO_ROOT / "results" / "epic05" / "phase7"

    # Available checkpoints (room only — bicycle/garden need server)
    # Focus on iter5000 (densification active) and iter30000 (final) — most informative
    checkpoints = {
        "room_t16_iter5000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter5000.pt"),
        "room_t16_iter30000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter30000.pt"),
        "room_t32_iter5000": str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter5000.pt"),
        "room_t32_iter30000": str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter30000.pt"),
    }

    viewmat, K, W, H = make_camera()

    results = {
        "experiment_id": "phase8-real-scene-microbench",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "config": {
            "resolution": f"{W}x{H}",
            "BATCH": BATCH,
            "N_REPEAT": N_REPEAT,
            "WARMUP": WARMUP,
        },
        "checkpoints": {},
    }

    for ckpt_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"SKIP: {ckpt_name} — file not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Loading: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"{'='*60}")

        params = load_checkpoint(ckpt_path)
        N = params["xyz"].shape[0]
        print(f"  Gaussians: {N}")
        print(f"  SH degree: {params['sh_degree']}")

        entry = {"num_gaussians": N, "sh_degree": params["sh_degree"], "tile_sizes": {}}

        for ts in [16, 32]:
            print(f"\n  --- tile_size={ts} ---")

            # Forward timing
            fwd_times, meta = timing_forward(params, viewmat, K, W, H, ts)
            fwd_mean = float(np.mean(fwd_times))
            fwd_std = float(np.std(fwd_times))
            fwd_cv = fwd_std / fwd_mean if fwd_mean > 0 else 0

            # Forward+Backward timing
            fwd_bwd_times, _ = timing_fwd_bwd(params, viewmat, K, W, H, ts)
            fwd_bwd_mean = float(np.mean(fwd_bwd_times))
            fwd_bwd_std = float(np.std(fwd_bwd_times))

            # Workload statistics
            wl_stats = compute_workload_stats(params, viewmat, K, W, H, ts)

            entry["tile_sizes"][str(ts)] = {
                "forward_ms_mean": fwd_mean,
                "forward_ms_std": fwd_std,
                "forward_ms_cv": fwd_cv,
                "fwd_bwd_ms_mean": fwd_bwd_mean,
                "fwd_bwd_ms_std": fwd_bwd_std,
                "workload_stats": wl_stats,
            }

            print(f"    Forward: {fwd_mean:.4f} ± {fwd_std:.4f} ms (CV={fwd_cv:.3f})")
            print(f"    Fwd+Bwd: {fwd_bwd_mean:.4f} ± {fwd_bwd_std:.4f} ms")
            print(f"    Tiles/Gaussian: {wl_stats['tiles_per_gauss_mean']:.2f} ± {wl_stats['tiles_per_gauss_std']:.2f}")
            print(f"    Total intersections: {wl_stats['total_intersections']:.0f}")

        # Compute ratio
        t16 = entry["tile_sizes"]["16"]
        t32 = entry["tile_sizes"]["32"]
        entry["ratio_t16_t32"] = {
            "forward": t16["forward_ms_mean"] / t32["forward_ms_mean"] if t32["forward_ms_mean"] > 0 else float("inf"),
            "fwd_bwd": t16["fwd_bwd_ms_mean"] / t32["fwd_bwd_ms_mean"] if t32["fwd_bwd_ms_mean"] > 0 else float("inf"),
        }
        print(f"\n  >>> Ratio t16/t32 forward: {entry['ratio_t16_t32']['forward']:.4f}")
        print(f"  >>> Ratio t16/t32 fwd+bwd: {entry['ratio_t16_t32']['fwd_bwd']:.4f}")

        results["checkpoints"][ckpt_name] = entry
        torch.cuda.empty_cache()

    # Save results
    output_path = output_dir / "real_scene_microbench.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
