#!/usr/bin/env python3
"""
Phase 7: Full 3DGS training with real GT.

Usage:
    python scripts/epic05/phase7/run_full.py \
        --scene room --tile-size 16 --steps 30000

    python scripts/epic05/phase7/run_full.py \
        --scene room --tile-size 32 --steps 30000

    python scripts/epic05/phase7/run_full.py \
        --scene room --tile-sizes 16 32 --steps 30000  (sequential)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.epic05.phase7.train_3dgs import TrainingConfig, TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Full 3DGS training")
    parser.add_argument("--scene", choices=["room", "garden", "bicycle"], default="room")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--packed", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.1)
    parser.add_argument("--lambda-dssim", type=float, default=0.2)
    parser.add_argument("--lr_xyz", type=float, default=1.6e-4)
    parser.add_argument("--lr_rotation", type=float, default=1e-3)
    parser.add_argument("--lr_scaling", type=float, default=5e-3)
    parser.add_argument("--lr_opacity", type=float, default=5e-2)
    parser.add_argument("--lr_sh", type=float, default=2.5e-3)
    parser.add_argument("--densification-start", type=int, default=500)
    parser.add_argument("--densification-end", type=int, default=15000)
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--grad-threshold", type=float, default=2e-4)
    parser.add_argument("--prune-interval", type=int, default=100)
    parser.add_argument("--prune-opacity", type=float, default=0.005)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint at given iteration ('latest' or number)")
    args = parser.parse_args()

    all_summaries = []
    for tile_size in args.tile_sizes:
        label = f"phase7_{args.scene}_t{tile_size}"
        if args.label:
            label = f"{args.label}_{tile_size}"

        config = TrainingConfig(
            scene=args.scene,
            resolution=args.resolution,
            repo_root=str(REPO_ROOT),
            num_iterations=args.steps,
            tile_size=tile_size,
            packed=args.packed,
            radius_clip=args.radius_clip,
            eps2d=args.eps2d,
            lambda_dssim=args.lambda_dssim,
            densification_start=args.densification_start,
            densification_end=args.densification_end,
            densification_interval=args.densification_interval,
            densification_grad_threshold=args.grad_threshold,
            prune_interval=args.prune_interval,
            prune_opacity_threshold=args.prune_opacity,
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
            experiment_label=label,
        )

        pipeline = TrainingPipeline(config, resume_from=args.resume)
        summary = pipeline.train()
        all_summaries.append(summary)

        print(f"\nSummary for tile_size={tile_size}:")
        print(f"  Total wall time: {summary['total_wall_s']:.1f}s")
        print(f"  Best PSNR: {summary['best_psnr']:.2f} dB")
        print(f"  Initial Gaussians: {summary['initial_gaussian_count']:,}")
        print(f"  Final Gaussians: {summary['final_gaussian_count']:,}")

    # Comparison summary if multiple tile sizes
    if len(all_summaries) >= 2:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        baseline = all_summaries[0]
        for s in all_summaries[1:]:
            speedup = baseline["total_wall_s"] / s["total_wall_s"] if s["total_wall_s"] > 0 else 1.0
            psnr_diff = s["best_psnr"] - baseline["best_psnr"]
            print(f"  t{baseline['tile_size']} vs t{s['tile_size']}:")
            print(f"    Wall time: {baseline['total_wall_s']:.0f}s vs {s['total_wall_s']:.0f}s = {speedup:.3f}x")
            print(f"    Best PSNR: {baseline['best_psnr']:.2f} vs {s['best_psnr']:.2f} (Δ={psnr_diff:.2f})")
            print(f"    Final N:   {baseline['final_gaussian_count']:,} vs {s['final_gaussian_count']:,}")

    # Save comparison
    if len(all_summaries) >= 2:
        comp_path = Path(REPO_ROOT) / "results" / "epic05" / "phase7" / f"comparison_{args.scene}_summary.json"
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_path, "w") as f:
            json.dump({"summaries": all_summaries, "comparison": {
                "baseline_tile": all_summaries[0]["tile_size"],
                "candidates": [s["tile_size"] for s in all_summaries[1:]],
            }}, f, indent=2)
        print(f"  Saved comparison: {comp_path}")


if __name__ == "__main__":
    main()
