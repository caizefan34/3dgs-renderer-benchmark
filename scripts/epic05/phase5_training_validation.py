#!/usr/bin/env python3
"""
Phase 5: Training Validation — Extended.

Compares tile16 vs tile32 during training on RTX 5070 Laptop.
Extends Phase 4 training to:
  - Multiple scenes (room, garden)
  - Full step decomposition (fwd/bwd/opt)
  - Per-stage analysis (early/middle/late)
  - Quality tracking (PSNR)
  - Gaussian count tracking

Usage:
    python scripts/epic05/phase5_training_validation.py --scene room --steps 5000
    python scripts/epic05/phase5_training_validation.py --scene garden --steps 5000
    python scripts/epic05/phase5_training_validation.py --scene room --tile-sizes 16 32 --steps 10000
"""
import argparse
import gc
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_MSVC_DIR = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
for _p in [_MSVC_DIR, _CUDA_BIN]:
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
import torch.utils.cpp_extension as cpp_ext
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from gsplat import rasterization


class SimpleGaussianModel(nn.Module):
    """Minimal 3DGS model for training comparison."""

    def __init__(self, num_points: int, sh_degree: int = 3, device="cuda"):
        super().__init__()
        self.num_points = num_points
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        self.xyz = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.01)
        self.rotations = nn.Parameter(
            torch.randn(num_points, 4, device=device) * 0.01
        )
        self.scales = nn.Parameter(
            torch.log(torch.ones(num_points, 3, device=device) * 0.01)
        )
        self.opacity = nn.Parameter(
            torch.logit(torch.full((num_points, 1), 0.5, device=device))
        )
        self.shs = nn.Parameter(
            torch.randn(num_points, self.num_sh_coeffs, 3, device=device) * 0.01
        )

    def forward(self):
        return {
            "xyz": self.xyz,
            "rotations": self.rotations,
            "scales": self.scales,
            "opacity": self.opacity,
            "shs": self.shs,
            "num_points": self.num_points,
        }


class DecomposedTrainer:
    """Trainer with decomposed timing for tile-size comparison."""

    def __init__(self, model: SimpleGaussianModel, tile_size: int = 16,
                 lr: float = 1e-3, device="cuda"):
        self.model = model.to(device)
        self.tile_size = tile_size
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self, camera, gt_image, track_gaussians: bool = False):
        """Run one training step with decomposed timing."""
        times = {}

        t0 = time.perf_counter()

        # Prepare scene data
        data = self.model.forward()
        quats = F.normalize(data["rotations"], dim=-1).contiguous()
        scales_activated = torch.exp(data["scales"]).contiguous()
        ops_activated = torch.sigmoid(data["opacity"]).squeeze(-1).contiguous()
        shs_cont = data["shs"].contiguous()

        torch.cuda.synchronize()
        t_prep = time.perf_counter()

        # Forward rasterization
        rendered, _, _ = rasterization(
            means=data["xyz"],
            quats=quats,
            scales=scales_activated,
            opacities=ops_activated,
            colors=shs_cont,
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width,
            height=camera.image_height,
            tile_size=self.tile_size,
            sh_degree=self.model.sh_degree,
            packed=True,
            render_mode="RGB",
        )
        rendered = rendered[0].clamp(0, 1)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        # Loss
        loss = F.mse_loss(rendered, gt_image)
        t_loss = time.perf_counter()

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter()

        # Optimizer step
        self.optimizer.step()
        t_opt = time.perf_counter()

        with torch.no_grad():
            psnr = 10 * torch.log10(1.0 / loss).item()

        result = {
            "loss": loss.item(),
            "psnr": psnr,
            "prep_ms": (t_prep - t0) * 1000,
            "forward_ms": (t_fwd - t_prep) * 1000,
            "backward_ms": (t_bwd - t_loss) * 1000,
            "optimizer_ms": (t_opt - t_bwd) * 1000,
            "total_ms": (t_opt - t0) * 1000,
        }

        if track_gaussians:
            result["num_gaussians"] = data["num_points"]

        return result


def create_random_gt(camera, device="cuda"):
    return torch.rand(
        (camera.image_height, camera.image_width, 3), device=device
    )


def run_training_validation(args):
    """Run training comparison between tile sizes with progress reporting."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "results" / "epic05" / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_path = REPO_ROOT / "data" / "official" / "mipnerf360" / args.scene / "point_cloud.ply"
    camera_path = REPO_ROOT / "data" / "official" / "mipnerf360" / args.scene / "cameras.json"

    print(f"  Loading scene: {scene_path}")
    scene_data = load_ply(str(scene_path), device="cuda")
    cameras = load_cameras_from_json(str(camera_path), device="cuda")

    RESOLUTION_PRESETS = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
    target_res = RESOLUTION_PRESETS.get(args.resolution, (1920, 1080))
    cameras = resize_cameras(cameras, *target_res)

    num_points = scene_data["num_points"]
    print(f"  Scene: {args.scene} ({num_points:,} gaussians)")
    print(f"  Resolution: {target_res[0]}x{target_res[1]}")
    print(f"  Steps: {args.steps}, Tile sizes: {args.tile_sizes}")
    print(f"  Log interval: {args.log_interval}")

    results = {}
    for tile_size in args.tile_sizes:
        print(f"\n{'='*60}")
        print(f"  Training with tile_size={tile_size}")
        print(f"{'='*60}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.manual_seed(args.seed)
        model = SimpleGaussianModel(
            num_points=num_points,
            sh_degree=3,
            device="cuda",
        )
        trainer = DecomposedTrainer(model, tile_size=tile_size, lr=args.lr)

        step_times: List[Dict] = []
        psnr_trajectory = []
        loss_trajectory = []
        stage_means: Dict[str, List[float]] = {
            "early": [], "middle": [], "late": []
        }

        # Use a fixed camera for consistency
        train_cam = cameras[0]
        gt_image = create_random_gt(train_cam)

        # Per-step tracking
        val_interval = max(1, args.steps // 20)  # ~20 checkpoints

        for step in range(args.steps):
            track_gaussians = (step % val_interval == 0)
            result = trainer.train_step(train_cam, gt_image,
                                        track_gaussians=track_gaussians)
            step_times.append(result)

            if step % val_interval == 0 or step == args.steps - 1:
                psnr_trajectory.append({"step": step, "psnr": result["psnr"]})
                loss_trajectory.append({"step": step, "loss": result["loss"]})
                gauss_str = (f" gauss={result.get('num_gaussians', num_points):,}"
                             if track_gaussians else "")
                print(f"    Step {step:5d}: total={result['total_ms']:7.2f}ms "
                      f"fwd={result['forward_ms']:7.2f}ms "
                      f"bwd={result['backward_ms']:7.2f}ms "
                      f"opt={result['optimizer_ms']:7.2f}ms "
                      f"PSNR={result['psnr']:6.2f}dB{gauss_str}")

            # Stage classification
            stage_idx = min(2, step * 3 // args.steps)
            stage_name = ["early", "middle", "late"][stage_idx]
            stage_means[stage_name].append(result["total_ms"])

        # Summary statistics
        total_times = np.array([r["total_ms"] for r in step_times])
        fwd_times = np.array([r["forward_ms"] for r in step_times])
        bwd_times = np.array([r["backward_ms"] for r in step_times])
        opt_times = np.array([r["optimizer_ms"] for r in step_times])

        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        tile_result = {
            "tile_size": tile_size,
            "num_steps": args.steps,
            "step_time_mean_ms": round(float(total_times.mean()), 4),
            "step_time_median_ms": round(float(np.median(total_times)), 4),
            "step_time_std_ms": round(float(total_times.std()), 4),
            "step_time_p25_ms": round(float(np.percentile(total_times, 25)), 4),
            "step_time_p75_ms": round(float(np.percentile(total_times, 75)), 4),
            "forward_mean_ms": round(float(fwd_times.mean()), 4),
            "backward_mean_ms": round(float(bwd_times.mean()), 4),
            "optimizer_mean_ms": round(float(opt_times.mean()), 4),
            "forward_pct": round(float(fwd_times.mean() / total_times.mean() * 100), 2),
            "backward_pct": round(float(bwd_times.mean() / total_times.mean() * 100), 2),
            "optimizer_pct": round(float(opt_times.mean() / total_times.mean() * 100), 2),
            "early_mean_ms": round(float(np.mean(stage_means["early"])), 4),
            "middle_mean_ms": round(float(np.mean(stage_means["middle"])), 4),
            "late_mean_ms": round(float(np.mean(stage_means["late"])), 4),
            "peak_vram_mb": round(peak_vram, 1),
            "final_psnr": float(psnr_trajectory[-1]["psnr"]) if psnr_trajectory else 0.0,
            "psnr_trajectory": psnr_trajectory,
            "loss_trajectory": loss_trajectory,
        }
        results[f"tile{tile_size}"] = tile_result

        print(f"\n  Summary tile{tile_size}:")
        print(f"    Mean step: {tile_result['step_time_mean_ms']:.2f}ms "
              f"(fwd={tile_result['forward_mean_ms']:.2f}ms "
              f"bwd={tile_result['backward_mean_ms']:.2f}ms "
              f"opt={tile_result['optimizer_mean_ms']:.2f}ms)")
        print(f"    Stages: early={tile_result['early_mean_ms']:.2f}ms "
              f"middle={tile_result['middle_mean_ms']:.2f}ms "
              f"late={tile_result['late_mean_ms']:.2f}ms")
        print(f"    VRAM: {tile_result['peak_vram_mb']:.0f}MB "
              f"Final PSNR: {tile_result['final_psnr']:.2f}dB")

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Speedup comparison
    print(f"\n{'='*60}")
    print(f"  Training Speedup Summary")
    print(f"{'='*60}")
    t16_mean = results.get("tile16", {}).get("step_time_mean_ms", None)
    for ts, rdata in results.items():
        t_this = rdata["step_time_mean_ms"]
        ratio = f"{t16_mean/t_this:.4f}x" if t16_mean and t_this else "N/A"
        print(f"  {ts}: {t_this:.2f}ms vs tile16: {t16_mean or 'N/A'}ms → {ratio}")

    # Save
    output_data = {
        "experiment_id": "epic05-phase5-training-v2",
        "date": date.today().isoformat(),
        "timestamp": timestamp,
        "gpu": torch.cuda.get_device_name(0),
        "scene": args.scene,
        "resolution": list(target_res),
        "num_gaussians": num_points,
        "training_config": {
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "tile_sizes": args.tile_sizes,
        },
        "results": results,
    }

    output_path = output_dir / f"training_{args.scene}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Training Validation (extended)")
    parser.add_argument("--scene", default="room",
                        choices=["room", "garden", "bicycle"],
                        help="Scene to train on")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", default="1080p")
    args = parser.parse_args()

    run_training_validation(args)


if __name__ == "__main__":
    main()
