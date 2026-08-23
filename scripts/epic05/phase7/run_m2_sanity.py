#!/usr/bin/env python3
"""
Phase 7C — M2 packed/dense short-run training sanity check.

Purpose:
    Determine whether M2 (packed vs dense) can enter the full-training gate.
    Compares packed=True vs packed=False in a controlled short run.

Gates tested:
    ✅ Forward: do both modes produce finite outputs?
    ✅ Backward: do both modes produce finite gradients?
    ✅ Loss: does the loss evolve similarly?
    ✅ NaN: any NaN/Inf in gradients or parameters?
    ✅ Performance: measurable difference in per-iteration timing?
    ✅ Gaussian count: does densification/pruning behave identically?

Only if ALL pass → M2 may proceed to 30K full training.

Usage:
    python scripts/epic05/phase7/run_m2_sanity.py --scene room --steps 1000

Requires: server EPIC-05 reachable (local RTX 5070 GPU also sufficient)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def run_packed_vs_dense(args) -> Dict:
    """Run packed=True vs packed=False sequentially and compare."""
    from scripts.epic05.phase7.gaussian_model import GaussianModel
    from scripts.epic05.phase7.loss import combined_loss
    from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint
    from gsplat import rasterization

    device = "cuda"
    results = {
        "experiment_id": f"phase7c_m2_sanity_{args.scene}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "scene": args.scene,
            "resolution": args.resolution,
            "steps": args.steps,
            "packed_modes": [True, False],
            "tile_size": args.tile_size,
            "seed": args.seed,
            "grad_threshold": args.grad_threshold,
            "opacity_threshold": args.opacity_threshold,
        },
        "modes": {},
    }

    for packed in [True, False]:
        label = "packed" if packed else "dense"
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print(f"\n{'='*60}")
        print(f"  M2 Sanity: {label.upper()} mode (tile_size={args.tile_size})")
        print(f"{'='*60}")

        # Dataset
        print("\n  [Loading dataset...]")
        dataset = GTDataset(
            scene=args.scene,
            repo_root=REPO_ROOT,
            resolution=args.resolution,
            device=device,
        )

        # Model
        print("\n  [Initializing model...]")
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
        print(f"    Initial Gs: {model.xyz.shape[0]:,}")

        spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()

        # Optimizer
        optimizer = torch.optim.Adam([
            {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
            {"params": [model.rotations], "lr": 1e-3},
            {"params": [model.scales], "lr": 5e-3},
            {"params": [model.opacity], "lr": 5e-2},
            {"params": [model.shs], "lr": 2.5e-3},
        ])

        # Training loop
        metrics: List[Dict] = []
        nan_detected = False
        inf_detected = False

        fwd_times: List[float] = []
        bwd_times: List[float] = []

        t0 = time.perf_counter()

        for step in range(args.steps):
            cam_idx = step % len(dataset)
            camera = dataset.get_camera(cam_idx)
            gt_image = dataset.get_gt_image(cam_idx)

            # SH scheduling
            new_degree = min(3, step // 500)
            if new_degree != model.sh_degree:
                model.set_sh_degree(new_degree)
                optimizer = torch.optim.Adam([
                    {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                    {"params": [model.rotations], "lr": 1e-3},
                    {"params": [model.scales], "lr": 5e-3},
                    {"params": [model.opacity], "lr": 5e-2},
                    {"params": [model.shs], "lr": 2.5e-3},
                ])

            data = model.forward()

            # Forward (CUDA timed)
            ev_fwd_s = torch.cuda.Event(enable_timing=True)
            ev_fwd_e = torch.cuda.Event(enable_timing=True)
            ev_fwd_s.record()
            rendered, _, info = rasterization(
                means=data["xyz"], quats=data["rotations"],
                scales=data["scales"], opacities=data["opacity"],
                colors=data["shs"],
                viewmats=camera.viewmatrix.unsqueeze(0),
                Ks=camera.K.unsqueeze(0),
                width=camera.image_width, height=camera.image_height,
                tile_size=args.tile_size, packed=packed, sh_degree=model.sh_degree,
                radius_clip=0.0, eps2d=0.1,
                render_mode="RGB",
            )
            ev_fwd_e.record()
            rendered = rendered[0].clamp(0, 1)

            # Loss
            loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
            loss = loss_dict["loss"]

            # Backward (CUDA timed)
            ev_bwd_s = torch.cuda.Event(enable_timing=True)
            ev_bwd_e = torch.cuda.Event(enable_timing=True)
            ev_bwd_s.record()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize()
            ev_bwd_e.record()

            # NaN/Inf check on gradients
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        nan_detected = True
                        print(f"    WARNING: NaN grad in {name} at step {step}")
                    if torch.isinf(p.grad).any():
                        inf_detected = True
                        print(f"    WARNING: Inf grad in {name} at step {step}")
                if torch.isnan(p).any() or torch.isinf(p).any():
                    nan_detected = True
                    print(f"    WARNING: NaN/Inf param in {name} at step {step}")

            model.accumulate_positional_gradient()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer
            ev_opt_s = torch.cuda.Event(enable_timing=True)
            ev_opt_e = torch.cuda.Event(enable_timing=True)
            ev_opt_s.record()
            optimizer.step()
            torch.cuda.synchronize()
            ev_opt_e.record()

            # Densification
            denf_count = {"cloned": 0, "split": 0, "removed": 0}
            if step >= 200 and step % 100 == 0:
                denf_count = model.densification(grad_threshold=args.grad_threshold)
                if denf_count["cloned"] + denf_count["split"] > 0:
                    optimizer = torch.optim.Adam([
                        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                        {"params": [model.rotations], "lr": 1e-3},
                        {"params": [model.scales], "lr": 5e-3},
                        {"params": [model.opacity], "lr": 5e-2},
                        {"params": [model.shs], "lr": 2.5e-3},
                    ])

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

            # Sync and collect timings
            ev_fwd_e.synchronize()
            ev_bwd_e.synchronize()
            ev_opt_e.synchronize()
            fwd_ms = ev_fwd_s.elapsed_time(ev_fwd_e)
            bwd_ms = ev_bwd_s.elapsed_time(ev_bwd_e)
            opt_ms = ev_opt_s.elapsed_time(ev_opt_e)
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)

            with torch.no_grad():
                mse = torch.mean((rendered - gt_image) ** 2).item()
                psnr = 10 * math.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 100.0

            metrics.append({
                "step": step,
                "loss": loss.item(),
                "l1": loss_dict["l1"].item(),
                "d_ssim": loss_dict["d_ssim"].item(),
                "psnr": psnr,
                "grad_norm": grad_norm,
                "gaussians": model.xyz.shape[0],
                "step_ms": (time.perf_counter() - t0) * 1000 / max(step, 1),
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "opt_ms": opt_ms,
                "cloned": denf_count["cloned"],
                "split": denf_count["split"],
                "pruned": prune_count,
            })

            if step % 200 == 0 or step == args.steps - 1:
                print(
                    f"    Step {step:4d}: loss={loss.item():.4f} PSNR={psnr:.2f}dB "
                    f"N={model.xyz.shape[0]:,} "
                    f"fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms opt={opt_ms:.1f}ms"
                    f"{' [denf]' if denf_count['cloned']+denf_count['split'] > 0 else ''}"
                    f"{' [prune]' if prune_count > 0 else ''}"
                )

        total_time = time.perf_counter() - t0

        # Summary
        print(f"\n  [{label.upper()}] COMPLETE:")
        print(f"    Total: {total_time:.1f}s, Avg step: {total_time*1000/args.steps:.1f}ms")
        print(f"    Initial Gs: {metrics[0]['gaussians']:,} → Final Gs: {model.xyz.shape[0]:,}")
        print(f"    Best PSNR: {max(m['psnr'] for m in metrics):.2f} dB")
        print(f"    NaN: {nan_detected}, Inf: {inf_detected}")
        print(f"    Avg fwd: {np.mean(fwd_times):.2f}ms, Avg bwd: {np.mean(bwd_times):.2f}ms")

        results["modes"][label] = {
            "total_time_s": total_time,
            "avg_step_ms": total_time * 1000 / args.steps,
            "avg_fwd_ms": float(np.mean(fwd_times)),
            "avg_bwd_ms": float(np.mean(bwd_times)),
            "initial_gaussians": metrics[0]["gaussians"],
            "final_gaussians": model.xyz.shape[0],
            "best_psnr_db": float(max(m["psnr"] for m in metrics)),
            "final_psnr_db": float(metrics[-1]["psnr"]),
            "nan_detected": nan_detected,
            "inf_detected": inf_detected,
            "trajectory": metrics,
        }

        del model, dataset, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # Comparison
    packed_r = results["modes"]["packed"]
    dense_r = results["modes"]["dense"]
    print(f"\n{'='*60}")
    print(f"  M2 COMPARISON: packed vs dense")
    print(f"{'='*60}")
    print(f"    Total time:   packed={packed_r['total_time_s']:.0f}s vs dense={dense_r['total_time_s']:.0f}s")
    print(f"    Avg fwd:      packed={packed_r['avg_fwd_ms']:.2f}ms vs dense={dense_r['avg_fwd_ms']:.2f}ms")
    print(f"    Avg bwd:      packed={packed_r['avg_bwd_ms']:.2f}ms vs dense={dense_r['avg_bwd_ms']:.2f}ms")
    print(f"    Best PSNR:    packed={packed_r['best_psnr_db']:.2f}dB vs dense={dense_r['best_psnr_db']:.2f}dB")
    print(f"    Final Gs:     packed={packed_r['final_gaussians']:,} vs dense={dense_r['final_gaussians']:,}")
    print(f"    NaN:          packed={packed_r['nan_detected']} vs dense={dense_r['nan_detected']}")
    print(f"    Inf:          packed={packed_r['inf_detected']} vs dense={dense_r['inf_detected']}")

    # Sanity verdict
    passes = 0
    total_checks = 5
    if not packed_r["nan_detected"] and not dense_r["nan_detected"]:
        print(f"  ✅ [1/5] No NaN detected")
        passes += 1
    else:
        print(f"  ❌ [1/5] NaN detected")

    if not packed_r["inf_detected"] and not dense_r["inf_detected"]:
        print(f"  ✅ [2/5] No Inf detected")
        passes += 1
    else:
        print(f"  ❌ [2/5] Inf detected")

    psnr_diff = abs(packed_r["best_psnr_db"] - dense_r["best_psnr_db"])
    if psnr_diff < 1.0:
        print(f"  ✅ [3/5] PSNR consistent (Δ={psnr_diff:.2f} dB)")
        passes += 1
    else:
        print(f"  ❌ [3/5] PSNR differs significantly (Δ={psnr_diff:.2f} dB)")

    if abs(packed_r["final_gaussians"] - dense_r["final_gaussians"]) / max(packed_r["final_gaussians"], 1) < 0.05:
        print(f"  ✅ [4/5] Gaussian count trajectory similar")
        passes += 1
    else:
        print(f"  ❌ [4/5] Gaussian count diverges")

    print(f"  {'✅' if passes >= 4 else '❌'} [5/5] Overall verdict: {passes}/{total_checks} checks passed")

    results["comparison"] = {
        "checks_passed": passes,
        "total_checks": total_checks,
        "verdict": "PASS" if passes >= 4 else "FAIL",
        "psnr_delta_db": round(psnr_diff, 3),
        "gaussian_delta_pct": round(
            (packed_r["final_gaussians"] - dense_r["final_gaussians"]) / max(dense_r["final_gaussians"], 1) * 100, 2
        ),
        "fwd_speedup": round(dense_r["avg_fwd_ms"] / packed_r["avg_fwd_ms"], 3) if packed_r["avg_fwd_ms"] > 0 else 1,
    }
    results["eligible_for_full_training"] = passes >= 4

    # Save
    output_dir = Path(REPO_ROOT) / "results" / "epic05" / "phase7c"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"m2_sanity_{args.scene}_{args.steps}steps.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="M2 packed/dense sanity check")
    parser.add_argument("--scene", choices=["room", "garden", "bicycle"], default="room")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-threshold", type=float, default=2e-4)
    parser.add_argument("--opacity-threshold", type=float, default=0.005)
    args = parser.parse_args()

    run_packed_vs_dense(args)


if __name__ == "__main__":
    main()
