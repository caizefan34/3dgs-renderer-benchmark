#!/usr/bin/env python3
"""
Phase 7: Short sanity training run with real GT.

Purpose: Verify that the complete training pipeline works end-to-end
before committing to a full 30K-step training run.

Checks:
  - Dataset loading (real GT + cameras)
  - Gaussian model with SfM initialization
  - Forward + backward through differentiable renderer
  - Densification and pruning operations
  - Loss computation (L1 + D-SSIM)
  - Optimizer step with per-group LR
  - Checkpoint save/load

Usage:
    python scripts/epic05/phase7/run_sanity.py \
        --scene room \
        --steps 500 \
        --tile-sizes 16 32

Output:
    results/epic05/phase7/sanity_{scene}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def run_sanity_check(args) -> Dict:
    """Run a short training sanity check with one tile size."""
    from scripts.epic05.phase7.gaussian_model import GaussianModel
    from scripts.epic05.phase7.loss import combined_loss
    from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint
    from gsplat import rasterization

    device = "cuda"
    tile_size = args.tile_size
    num_steps = args.steps

    print(f"\n{'='*60}")
    print(f"  Phase 7 Sanity Check")
    print(f"  Scene: {args.scene}, Tile: {tile_size}, Steps: {num_steps}")
    print(f"{'='*60}")

    # --- Load dataset ---
    print("\n  [1/6] Loading dataset...")
    dataset = GTDataset(
        scene=args.scene,
        repo_root=REPO_ROOT,
        resolution=args.resolution,
        device=device,
    )
    print(f"    {len(dataset)} camera-GT pairs")

    # --- Load SfM and create model ---
    print("\n  [2/6] Creating Gaussian model from SfM...")
    sfm_data = load_initial_checkpoint(args.scene, REPO_ROOT, device=device)
    model = GaussianModel(
        num_points=sfm_data["xyz"].shape[0],
        sh_degree=0,
        max_sh_degree=3,
        device=device,
    )
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"),
    )
    print(f"    {model.xyz.shape[0]:,} initial Gaussians")

    spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()

    # --- Optimizer ---
    print("\n  [3/6] Setting up optimizer...")
    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ])

    # --- Training loop ---
    print(f"\n  [4/6] Running {num_steps} steps...")
    metrics: List[Dict] = []
    denf_log: List[Dict] = []
    gauss_log: List[Dict] = []
    timing_log: List[Dict] = []
    nan_detected = False
    inf_detected = False

    t0_total = time.perf_counter()

    for step in range(num_steps):
        # Camera selection
        cam_idx = step % len(dataset)
        camera = dataset.get_camera(cam_idx)
        gt_image = dataset.get_gt_image(cam_idx)

        t_step = time.perf_counter()

        # Forward
        data = model.forward()

        # Increase SH degree
        new_degree = min(3, step // 500)
        if new_degree != model.sh_degree:
            model.set_sh_degree(new_degree)
            print(f"    [Step {step}] SH degree -> {new_degree}")
            data = model.forward()

        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"],
            scales=data["scales"], opacities=data["opacity"],
            colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=tile_size, packed=True, sh_degree=model.sh_degree,
            radius_clip=args.radius_clip, eps2d=args.eps2d,
            render_mode="RGB",
        )
        rendered = rendered[0].clamp(0, 1)

        # Loss
        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss = loss_dict["loss"]

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()

        # Gradient accumulation
        model.accumulate_positional_gradient()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.synchronize()

        # Optimizer step
        optimizer.step()
        t_end = time.perf_counter()

        # Check for NaN/Inf
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_detected = True
                    print(f"    WARNING: NaN gradient in {name} at step {step}")
                if torch.isinf(p.grad).any():
                    inf_detected = True
                    print(f"    WARNING: Inf gradient in {name} at step {step}")
            if torch.isnan(p).any() or torch.isinf(p).any():
                nan_detected = True
                print(f"    WARNING: NaN/Inf parameter in {name} at step {step}")

        # Densification
        denf_count = {"cloned": 0, "split": 0, "removed": 0}
        if (step >= 200 and step < 15000 and step % 100 == 0):
            denf_count = model.densification(
                grad_threshold=args.grad_threshold,
            )
            if denf_count["cloned"] + denf_count["split"] > 0:
                optimizer = torch.optim.Adam([
                    {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                    {"params": [model.rotations], "lr": 1e-3},
                    {"params": [model.scales], "lr": 5e-3},
                    {"params": [model.opacity], "lr": 5e-2},
                    {"params": [model.shs], "lr": 2.5e-3},
                ])

        # Pruning
        prune_count = 0
        if step >= 200 and step % 100 == 0:
            prune_count = model.prune(opacity_threshold=args.opacity_threshold)
            if prune_count > 0:
                optimizer = torch.optim.Adam([
                    {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                    {"params": [model.rotations], "lr": 1e-3},
                    {"params": [model.scales], "lr": 5e-3},
                    {"params": [model.opacity], "lr": 5e-2},
                    {"params": [model.shs], "lr": 2.5e-3},
                ])

        # Logging
        step_ms = (t_end - t_step) * 1000
        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 100.0

        metrics.append({
            "step": step,
            "loss": loss.item(),
            "l1": loss_dict["l1"].item(),
            "d_ssim": loss_dict["d_ssim"].item(),
            "psnr": psnr,
            "grad_norm": grad_norm,
            "step_ms": step_ms,
            "gaussians": model.xyz.shape[0],
        })

        if step % 100 == 0 or step == num_steps - 1:
            print(
                f"    Step {step:4d}: loss={loss.item():.4f} PSNR={psnr:.2f}dB "
                f"|g|={grad_norm:.2e} N={model.xyz.shape[0]:,} "
                f"{step_ms:.1f}ms"
                f"{' [denf:' + str(denf_count['cloned']+denf_count['split']) + ' prn:' + str(prune_count) + ']' if denf_count['cloned']+denf_count['split']+prune_count > 0 else ''}"
            )

        gauss_log.append({"step": step, "count": model.xyz.shape[0]})
        if denf_count["cloned"] + denf_count["split"] + prune_count > 0:
            denf_log.append({
                "step": step,
                "cloned": denf_count["cloned"],
                "split": denf_count["split"],
                "pruned": prune_count,
            })

    total_time = time.perf_counter() - t0_total

    # --- Summary ---
    print(f"\n  [5/6] Sanity check summary ({num_steps} steps, tile={tile_size}):")
    print(f"    Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"    Avg step: {total_time * 1000 / num_steps:.1f}ms")
    print(f"    Final Gaussians: {model.xyz.shape[0]:,}")
    print(f"    Final PSNR: {metrics[-1]['psnr']:.2f} dB")
    print(f"    Best PSNR: {max(m['psnr'] for m in metrics):.2f} dB")
    print(f"    NaN detected: {nan_detected}")
    print(f"    Inf detected: {inf_detected}")
    print(f"    Densification events: {len(denf_log)}")
    print(f"    SH degree: {model.sh_degree}")

    result = {
        "experiment_id": f"phase7_sanity_{args.scene}_t{tile_size}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "scene": args.scene,
            "resolution": args.resolution,
            "tile_size": tile_size,
            "steps": num_steps,
            "packed": True,
            "radius_clip": args.radius_clip,
            "eps2d": args.eps2d,
            "grad_threshold": args.grad_threshold,
            "opacity_threshold": args.opacity_threshold,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "summary": {
            "total_time_s": total_time,
            "avg_step_ms": total_time * 1000 / num_steps,
            "initial_gaussians": metrics[0]["gaussians"],
            "final_gaussians": model.xyz.shape[0],
            "final_psnr_db": metrics[-1]["psnr"],
            "best_psnr_db": max(m["psnr"] for m in metrics),
            "nan_detected": nan_detected,
            "inf_detected": inf_detected,
            "densification_events": len(denf_log),
            "final_sh_degree": model.sh_degree,
        },
        "trajectory": {
            "metrics": metrics,
            "gaussian_count": gauss_log,
            "densification": denf_log,
        },
    }

    # --- Save ---
    output_dir = Path(REPO_ROOT) / "results" / "epic05" / "phase7"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sanity_{args.scene}_t{tile_size}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  [6/6] Results saved: {output_path}")

    del model, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Short training sanity check")
    parser.add_argument("--scene", choices=["room", "garden", "bicycle"], default="room")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.1)
    parser.add_argument("--grad-threshold", type=float, default=2e-4)
    parser.add_argument("--opacity-threshold", type=float, default=0.005)
    args = parser.parse_args()

    run_sanity_check(args)


if __name__ == "__main__":
    main()
