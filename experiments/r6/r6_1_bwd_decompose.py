#!/usr/bin/env python3
"""
R6-1: Backward wall-time decomposition profiler.

Loads a baseline checkpoint, replays forward+backward, and decomposes:
  T_bwd = T_alloc/zero + T_raster + T_post (SH/projection/sigmoid)

Uses:
  - CUDA Events for total T_bwd wall time
  - torch.profiler (CUDA activity) for per-kernel decomposition
  - Intersection metadata for N_total / N_visible / N_touched

Usage (on mx):
  CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/r6_1_bwd_decompose.py \
    --scene room --ckpt /mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_5000.pt \
    --output /mnt/storage_pool/liaoyuanjun/r6_profiling/r6_1_room_5k.json
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, time, argparse, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
# Add baseline/reference_v1 LAST so it takes import priority (insert at 0)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization
from dataset import GTDataset
from trainer import SepSSIM, render_with_meta


def load_model_from_ckpt(ckpt_path, config, repo_root):
    """Restore a GaussianModel from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    model.active_sh_degree = ckpt["active_sh_degree"]
    return model, ckpt["num_points"]


def stats_ms(values):
    """Compute timing statistics in milliseconds."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std()),
        "n": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def profile_backward_decomposition(model, dataset, cam, gt_image, ssim_fn,
                                    sh_degree, n_warmup=20, n_measure=100):
    """Profile backward with CUDA events + torch.profiler."""

    # === CUDA Events for total T_bwd ===
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    bwd_times = []
    fwd_times = []
    iter_times = []

    # === Warmup ===
    for _ in range(n_warmup):
        model.optimizer.zero_grad(set_to_none=True)
        image, meta, means2d = render_with_meta(model, cam, sh_degree)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - 0.2) * L1 + 0.2 * dssim
        loss.backward()
        torch.cuda.synchronize()

    # === Measure total T_bwd with CUDA events ===
    for _ in range(n_measure):
        model.optimizer.zero_grad(set_to_none=True)

        iter_start.record()
        fwd_start.record()
        image, meta, means2d = render_with_meta(model, cam, sh_degree)
        fwd_end.record()
        torch.cuda.synchronize()
        fwd_ms = fwd_start.elapsed_time(fwd_end)

        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - 0.2) * L1 + 0.2 * dssim

        bwd_start.record()
        loss.backward()
        bwd_end.record()
        torch.cuda.synchronize()
        bwd_ms = bwd_start.elapsed_time(bwd_end)

        iter_end.record()
        torch.cuda.synchronize()
        iter_ms = iter_start.elapsed_time(iter_end)

        bwd_times.append(bwd_ms)
        fwd_times.append(fwd_ms)
        iter_times.append(iter_ms)

    # === torch.profiler for kernel-level decomposition (BACKWARD ONLY) ===
    # Profile ONLY loss.backward() — forward is pre-computed outside the profiler
    # context so the decomposition captures only backward kernels + memset.
    from torch.profiler import profile, ProfilerActivity

    n_prof = 30
    kernel_categories = {
        "raster_bwd": [],
        "sh_bwd": [],
        "proj_bwd": [],
        "memset_zero": [],
        "other_bwd": [],
    }

    for _ in range(n_prof):
        model.optimizer.zero_grad(set_to_none=True)
        image, meta, means2d = render_with_meta(model, cam, sh_degree)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - 0.2) * L1 + 0.2 * dssim
        # Profile ONLY the backward call
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            loss.backward()
            torch.cuda.synchronize()
        for evt in prof.key_averages():
            name = evt.key
            cuda_self = evt.self_device_time_total  # microseconds
            if cuda_self == 0:
                continue
            name_lower = name.lower()
            if "rasterize_to_pixels_3dgs_bwd" in name_lower:
                kernel_categories["raster_bwd"].append(cuda_self)
            elif "spherical_harmonics" in name_lower and "bwd" in name_lower:
                kernel_categories["sh_bwd"].append(cuda_self)
            elif "projection_ewa" in name_lower and "bwd" in name_lower:
                kernel_categories["proj_bwd"].append(cuda_self)
            elif "memset" in name_lower or "fill" in name_lower:
                kernel_categories["memset_zero"].append(cuda_self)
            else:
                kernel_categories["other_bwd"].append(cuda_self)

    # Sum per-category (microseconds → milliseconds), divide by n_profiled iters
    kernel_summary = {}
    for cat, times_us in kernel_categories.items():
        total_us = sum(times_us)
        kernel_summary[cat] = {
            "total_ms": total_us / 1e3,
            "per_iter_ms": total_us / n_prof / 1e3,
            "n_entries": len(times_us),
        }

    # Backward decomposition: T_bwd ≈ T_raster + T_zero + T_sh + T_proj + T_other
    T_raster = kernel_summary["raster_bwd"]["per_iter_ms"]
    T_zero = kernel_summary["memset_zero"]["per_iter_ms"]
    T_sh = kernel_summary["sh_bwd"]["per_iter_ms"]
    T_proj = kernel_summary["proj_bwd"]["per_iter_ms"]
    T_other_bwd = kernel_summary["other_bwd"]["per_iter_ms"]
    T_bwd_decomposed = T_raster + T_zero + T_sh + T_proj + T_other_bwd
    kernel_summary["T_bwd_decomposed_ms"] = T_bwd_decomposed
    kernel_summary["T_zero_pct_bwd"] = T_zero / max(T_bwd_decomposed, 1e-6) * 100
    kernel_summary["T_raster_pct_bwd"] = T_raster / max(T_bwd_decomposed, 1e-6) * 100

    # === Workload statistics ===
    N_total = model._xyz.shape[0]
    radii = meta["radii"][0]  # [N, 2]
    visibility_filter = (radii > 0).any(dim=-1)  # [N]
    N_visible = int(visibility_filter.sum().item())

    # N_touched: unique Gaussian IDs in the intersection list
    flatten_ids = meta.get("flatten_ids")
    if flatten_ids is not None:
        # flatten_ids is [n_isects] — unique Gaussian IDs that intersect tiles
        unique_ids = torch.unique(flatten_ids)
        N_intersected = int(unique_ids.shape[0])
    else:
        N_intersected = N_visible

    # Gradient buffer sizes (bytes)
    K_active = (sh_degree + 1) ** 2
    grad_buffer_bytes = {
        "rasterizer": {
            "v_means2d": N_total * 2 * 4,
            "v_conics": N_total * 3 * 4,
            "v_colors": N_total * 3 * 4,
            "v_opacities": N_total * 4,
            "v_means2d_abs": N_total * 2 * 4,
            "total": N_total * 11 * 4,
        },
        "projection": {
            "v_means": N_total * 3 * 4,
            "v_quats": N_total * 4 * 4,
            "v_scales": N_total * 3 * 4,
            "total": N_total * 10 * 4,
        },
        "sh": {
            "v_coefficients": N_total * K_active * 3 * 4,
            "total": N_total * K_active * 3 * 4,
        },
    }
    grad_buffer_bytes["grand_total"] = (
        grad_buffer_bytes["rasterizer"]["total"] +
        grad_buffer_bytes["projection"]["total"] +
        grad_buffer_bytes["sh"]["total"]
    )

    return {
        "scene": None,  # filled by caller
        "checkpoint": None,
        "N_total": N_total,
        "N_visible": N_visible,
        "N_intersected": N_intersected,
        "r_touch": N_intersected / max(N_total, 1),
        "sh_degree": sh_degree,
        "K_active": K_active,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "T_bwd_ms": stats_ms(bwd_times),
        "T_fwd_ms": stats_ms(fwd_times),
        "T_iter_ms": stats_ms(iter_times),
        "kernel_decomposition": kernel_summary,
        "grad_buffer_bytes": grad_buffer_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--camera-idx", type=int, default=0,
                        help="Camera to profile (default: 0)")
    args = parser.parse_args()

    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = str(REPO_ROOT)
    config.resolution = args.resolution
    config.seed = args.seed

    print(f"=== R6-1 Backward Decomposition: {args.scene} ===")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Camera: {args.camera_idx}")

    # Load dataset
    dataset = GTDataset(
        scene=config.scene, repo_root=config.repo_root,
        resolution=config.resolution, device="cuda", background="black",
    )
    print(f"  {len(dataset)} cameras loaded")

    # Load model
    model, initial_N = load_model_from_ckpt(args.ckpt, config, config.repo_root)
    print(f"  Model loaded: N={model._xyz.shape[0]}, SH degree={model.active_sh_degree}")

    # Get camera + GT
    cam, gt_image = dataset.get_item(args.camera_idx)
    ssim_fn = SepSSIM(device="cuda")

    # Profile
    result = profile_backward_decomposition(
        model, dataset, cam, gt_image, ssim_fn,
        model.active_sh_degree, args.n_warmup, args.n_measure,
    )
    result["scene"] = args.scene
    result["checkpoint"] = args.ckpt
    result["camera_idx"] = args.camera_idx
    result["image_resolution"] = [cam.image_width, cam.image_height]

    # Print summary
    print(f"\n=== Results: {args.scene} ===")
    print(f"  N_total={result['N_total']}, N_visible={result['N_visible']}, "
          f"N_intersected={result['N_intersected']}, r_touch={result['r_touch']:.4f}")
    print(f"  T_bwd: mean={result['T_bwd_ms']['mean']:.2f}ms, "
          f"median={result['T_bwd_ms']['median']:.2f}ms, "
          f"p95={result['T_bwd_ms']['p95']:.2f}ms, std={result['T_bwd_ms']['std']:.2f}ms")
    print(f"  T_fwd: mean={result['T_fwd_ms']['mean']:.2f}ms")
    print(f"  T_iter: mean={result['T_iter_ms']['mean']:.2f}ms")
    print(f"  Kernel decomposition (per-iter ms, backward-only):")
    for cat, s in result["kernel_decomposition"].items():
        if isinstance(s, dict) and s.get("per_iter_ms", 0) > 0:
            print(f"    {cat:20s}: {s['per_iter_ms']:.3f}ms ({s['n_entries']} entries/{s['total_ms']:.1f}ms total)")
    print(f"  Grad buffer bytes: {result['grad_buffer_bytes']['grand_total'] / 1e6:.1f} MB")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
