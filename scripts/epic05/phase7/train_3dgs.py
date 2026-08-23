"""
Phase 7 Training Pipeline — Full 3DGS training with real GT.

Strict mode: no shortcuts, no random GT, no simplified loops.
Implements proper 3DGS training with:
  - Real GT images from Mip-NeRF 360
  - L1 + D-SSIM loss
  - Densification (gradient-threshold clone/split)
  - Pruning (opacity-based)
  - SH degree progressive scheduling
  - Per-group Adam with original 3DGS learning rates
  - Checkpointing
  - Full metrics logging
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from benchmark_framework import load_ply  # noqa: E402
from gsplat import rasterization  # noqa: E402

from .gaussian_model import GaussianModel  # noqa: E402
from .loss import combined_loss  # noqa: E402
from .dataset import GTDataset, load_initial_checkpoint  # noqa: E402
from .checkpoint import TrainingCheckpoint  # noqa: E402


@dataclass
class TrainingConfig:
    """Phase 7 training configuration."""

    # Scene
    scene: str = "room"
    resolution: str = "1080p"
    repo_root: str = ""

    # Training steps
    num_iterations: int = 30000
    val_interval: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 5000

    # Renderer
    renderer: str = "gsplat"
    tile_size: int = 16
    packed: bool = True
    sh_degree: int = 3
    radius_clip: float = 0.0
    eps2d: float = 0.1

    # Loss
    lambda_dssim: float = 0.2

    # Learning rates (original 3DGS defaults)
    lr_xyz: float = 1.6e-4
    lr_rotation: float = 1e-3
    lr_scaling: float = 5e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3

    # Spatial LR scaling
    spatial_lr_scale: float = 1.0

    # Densification
    densification_interval: int = 100
    densification_grad_threshold: float = 0.0002
    densification_start: int = 500
    densification_end: int = 15000
    clone_max_screen_size: float = 100.0
    split_max_screen_size: float = 100.0

    # Pruning
    prune_interval: int = 100
    prune_opacity_threshold: float = 0.005
    prune_start: int = 500
    reset_opacity_interval: int = 3000

    # SH degree scheduling
    sh_degree_interval: int = 1000  # Increase SH degree every N steps from 0
    max_sh_degree: int = 3

    # Optimizer
    adam_eps: float = 1e-15
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Checkpoint
    save_dir: str = ""

    # Naming
    experiment_label: str = "phase7_room_t16"
    seed: int = 42


class TrainingPipeline:
    """Complete 3DGS training pipeline with real GT."""

    def __init__(self, config: TrainingConfig, resume_from: Optional[str | int] = None):
        self.config = config
        self._resume_from = resume_from
        self._resume_iteration = 0
        self.device = "cuda"

        # Resolve paths
        if not config.repo_root:
            config.repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        if not config.save_dir:
            config.save_dir = str(Path(config.repo_root) / "results" / "epic05" / "phase7")

        self.repo_root = Path(config.repo_root)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        mode = "RESUME" if resume_from else "NEW"
        print(f"  Phase 7 Training [{mode}] — {config.experiment_label}")
        print(f"  Scene: {config.scene}, Resolution: {config.resolution}")
        print(f"  Tile size: {config.tile_size}, Iterations: {config.num_iterations}")
        print(f"{'='*60}")

        # Load dataset
        print("\n  [Loading dataset...]")
        self.dataset = GTDataset(
            scene=config.scene,
            repo_root=self.repo_root,
            resolution=config.resolution,
            device=self.device,
        )

        # Checkpoint helper
        self.checkpointer = TrainingCheckpoint(self.save_dir / config.experiment_label)

        if resume_from is not None:
            # ---- RESUME from checkpoint ----
            resume_iter = int(resume_from) if isinstance(resume_from, (int, str)) and str(resume_from).isdigit() else None
            print(f"\n  [Resuming from checkpoint (iteration={resume_iter or 'latest'})...]")
            ckpt = self.checkpointer.load(iteration=resume_iter, label=config.experiment_label)
            if ckpt is None:
                raise FileNotFoundError(f"No checkpoint found to resume from (iteration={resume_iter})")
            ckpt_iter = ckpt["iteration"]
            print(f"    Found checkpoint at iteration {ckpt_iter}")

            self.model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=self.device)
            self.spatial_lr_scale = ckpt.get("spatial_lr_scale", 46.64)
            self._resume_iteration = ckpt_iter
            self._best_psnr = ckpt.get("metrics", {}).get("best_psnr", 0.0)

            # Rebuild optimizer (must match current param shapes)
            self.optimizer = torch.optim.Adam([
                {"params": [self.model.xyz], "lr": config.lr_xyz * self.spatial_lr_scale,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.rotations], "lr": config.lr_rotation,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.scales], "lr": config.lr_scaling,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.opacity], "lr": config.lr_opacity,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.shs], "lr": config.lr_sh,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
            ])
            if "optimizer_state" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])

            print(f"    Model: {self.model.xyz.shape[0]:,} Gaussians, SH deg={self.model.sh_degree}")
            print(f"    Resume from iteration {self._resume_iteration}")
            torch.cuda.empty_cache()
        else:
            # ---- NEW training from SfM ----
            print("\n  [Loading SfM initialization...]")
            sfm_data = load_initial_checkpoint(config.scene, self.repo_root, device="cuda")

            print("\n  [Initializing Gaussian model...]")
            self.model = GaussianModel(
                num_points=sfm_data["xyz"].shape[0],
                sh_degree=0,
                max_sh_degree=config.max_sh_degree,
                device=self.device,
            )
            self.model.init_from_sfm(
                xyz=sfm_data["xyz"],
                opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=self.device)),
                scales_log=sfm_data.get("scales"),
                rotations_raw=sfm_data.get("rotations"),
                shs=sfm_data.get("shs"),
            )

            scene_extent = sfm_data["xyz"].norm(dim=-1).max().item()
            self.spatial_lr_scale = scene_extent

            print(f"    SfM points: {self.model.xyz.shape[0]:,}")
            print(f"    Scene extent: {scene_extent:.2f}")

            self.optimizer = torch.optim.Adam([
                {"params": [self.model.xyz], "lr": config.lr_xyz * self.spatial_lr_scale,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.rotations], "lr": config.lr_rotation,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.scales], "lr": config.lr_scaling,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.opacity], "lr": config.lr_opacity,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
                {"params": [self.model.shs], "lr": config.lr_sh,
                 "eps": config.adam_eps, "betas": (config.adam_beta1, config.adam_beta2)},
            ])

        # Loss tracking
        self.metrics_log: List[Dict] = []
        self.psnr_log: List[Dict] = []
        self.loss_log: List[Dict] = []
        self.gaussian_log: List[Dict] = []
        self.denf_log: List[Dict] = []
        self.timing_log: List[Dict] = []

        self._iteration = 0
        if not resume_from:
            self._best_psnr = 0.0

    def train(self):
        """Run the complete training loop."""
        import gc
        config = self.config
        num_cameras = len(self.dataset)

        remaining = config.num_iterations - self._resume_iteration
        print(f"\n  [Starting training loop — {remaining} remaining iterations]")
        print(f"  Cameras: {num_cameras}, Densification: iterations "
              f"{config.densification_start}-{config.densification_end}")
        print(f"  Resume from iteration {self._resume_iteration}")
        print()

        start_time = time.perf_counter()

        # Pre-allocate CUDA events for per-stage timing
        _ev_fwd_start = torch.cuda.Event(enable_timing=True)
        _ev_fwd_end = torch.cuda.Event(enable_timing=True)
        _ev_bwd_start = torch.cuda.Event(enable_timing=True)
        _ev_bwd_end = torch.cuda.Event(enable_timing=True)
        _ev_opt_start = torch.cuda.Event(enable_timing=True)
        _ev_opt_end = torch.cuda.Event(enable_timing=True)
        _timing_ev_count = 0

        for iteration in range(self._resume_iteration + 1, config.num_iterations):
            self._iteration = iteration
            iteration_start = time.perf_counter()

            # --- SH degree scheduling ---
            new_degree = min(
                config.max_sh_degree,
                iteration // config.sh_degree_interval
            )
            if new_degree != self.model.sh_degree and new_degree <= config.max_sh_degree:
                self.model.set_sh_degree(new_degree)
                self._reconfigure_optimizer()
                print(f"    [Step {iteration}] SH degree increased to {new_degree}")

            # --- Select camera (round-robin) ---
            cam_idx = iteration % num_cameras
            camera = self.dataset.get_camera(cam_idx)
            gt_image = self.dataset.get_gt_image(cam_idx)

            # --- Forward (CUDA timed) ---
            data = self.model.forward()
            _ev_fwd_start.record()
            rendered, _, info = rasterization(
                means=data["xyz"],
                quats=data["rotations"],
                scales=data["scales"],
                opacities=data["opacity"],
                colors=data["shs"],
                viewmats=camera.viewmatrix.unsqueeze(0),
                Ks=camera.K.unsqueeze(0),
                width=camera.image_width,
                height=camera.image_height,
                tile_size=config.tile_size,
                packed=config.packed,
                sh_degree=self.model.sh_degree,
                radius_clip=config.radius_clip,
                eps2d=config.eps2d,
                render_mode="RGB",
                sparse_grad=False,
                absgrad=False,
            )
            _ev_fwd_end.record()
            rendered = rendered[0].clamp(0, 1)

            # --- Loss ---
            loss_dict = combined_loss(rendered, gt_image, lambda_dssim=config.lambda_dssim)
            loss = loss_dict["loss"]
            loss_item = loss.item()
            l1_item = loss_dict["l1"].item()
            d_ssim_item = loss_dict["d_ssim"].item()

            with torch.no_grad():
                mse = torch.mean((rendered - gt_image) ** 2).item()
                psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

            # --- Backward (CUDA timed) ---
            _ev_bwd_start.record()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize()
            _ev_bwd_end.record()

            # Accumulate positional gradient for densification
            self.model.accumulate_positional_gradient()

            # Gradient clipping (global norm)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            # --- Optimizer step (CUDA timed) ---
            _ev_opt_start.record()
            self.optimizer.step()
            torch.cuda.synchronize()
            _ev_opt_end.record()

            iteration_ms = (time.perf_counter() - iteration_start) * 1000

            # --- Densification & Pruning (topology change AFTER optimizer step) ---
            denf_start = time.perf_counter()
            denf_count = {"cloned": 0, "split": 0, "removed": 0}
            if (iteration >= config.densification_start
                    and iteration < config.densification_end
                    and iteration % config.densification_interval == 0):
                denf_count = self.model.densification(
                    grad_threshold=config.densification_grad_threshold,
                    clone_max_screen_size=config.clone_max_screen_size,
                    split_max_screen_size=config.split_max_screen_size,
                )

            prune_count = 0
            if (iteration >= config.prune_start
                    and iteration % config.prune_interval == 0):
                prune_count = self.model.prune_and_reset(
                    opacity_threshold=config.prune_opacity_threshold,
                    reset_interval=config.reset_opacity_interval,
                    current_step=iteration,
                )

            # Rebuild optimizer for next iteration if topology changed
            if denf_count["cloned"] + denf_count["split"] + prune_count > 0:
                self._reconfigure_optimizer()

            topology_ms = (time.perf_counter() - denf_start) * 1000

            # --- Peak memory ---
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            # --- CUDA event sync & timing ---
            _ev_fwd_end.synchronize()
            fwd_ms = _ev_fwd_start.elapsed_time(_ev_fwd_end)
            _ev_bwd_end.synchronize()
            bwd_ms = _ev_bwd_start.elapsed_time(_ev_bwd_end)
            _ev_opt_end.synchronize()
            opt_ms = _ev_opt_start.elapsed_time(_ev_opt_end)

            # --- Logging ---
            total_denf = denf_count["cloned"] + denf_count["split"] + denf_count["removed"]

            if iteration % config.log_interval == 0:
                self.metrics_log.append({
                    "iteration": iteration,
                    "loss": loss_item,
                    "l1": l1_item,
                    "d_ssim": d_ssim_item,
                    "psnr": psnr,
                    "grad_norm": grad_norm,
                    "num_gaussians": self.model.xyz.shape[0],
                    "iteration_ms": iteration_ms,
                    "fwd_ms": fwd_ms,
                    "bwd_ms": bwd_ms,
                    "opt_ms": opt_ms,
                    "topology_ms": topology_ms,
                    "peak_memory_mb": peak_memory_mb,
                    "current_memory_mb": current_memory_mb,
                    "cloned": denf_count["cloned"],
                    "split": denf_count["split"],
                    "pruned": prune_count,
                    "sh_degree": self.model.sh_degree,
                })

            if iteration % config.val_interval == 0 or iteration == config.num_iterations - 1:
                print(
                    f"  [{iteration:5d}/{config.num_iterations}] "
                    f"loss={loss_item:.4f} l1={l1_item:.4f} d-ssim={d_ssim_item:.4f} "
                    f"PSNR={psnr:.2f} |g|={grad_norm:.2e} "
                    f"N={self.model.xyz.shape[0]:,} "
                    f"t={iteration_ms:.1f}ms "
                    f"mem={peak_memory_mb:.0f}MB"
                )
                if total_denf > 0 or prune_count > 0:
                    print(
                        f"         (denf: clone={denf_count['cloned']}, "
                        f"split={denf_count['split']}, "
                        f"prune={prune_count})"
                    )

                self.psnr_log.append({"iteration": iteration, "psnr": psnr})
                self.loss_log.append({"iteration": iteration, "loss": loss_item})
                self.gaussian_log.append({"iteration": iteration, "count": self.model.xyz.shape[0]})

                # Track best PSNR
                if psnr > self._best_psnr:
                    self._best_psnr = psnr

            # --- Checkpoint ---
            if iteration > 0 and iteration % config.checkpoint_interval == 0:
                self._save_checkpoint(iteration, {"psnr": psnr, "loss": loss_item})

            # ── Periodic cleanup ──
            if iteration % 500 == 0:
                gc.collect()

        # --- Training complete ---
        total_wall_s = time.perf_counter() - start_time
        print(f"\n  [Training complete]")
        print(f"  Total time: {total_wall_s:.1f}s ({total_wall_s / 60:.1f} min)")
        print(f"  Best PSNR: {self._best_psnr:.2f} dB")
        print(f"  Final Gaussian count: {self.model.xyz.shape[0]:,}")
        print(f"  SH degree: {self.model.sh_degree}")

        # Save final checkpoint
        self._save_checkpoint(config.num_iterations, {
            "psnr": psnr,
            "loss": loss_item,
            "total_wall_s": total_wall_s,
            "best_psnr": self._best_psnr,
            "final_gaussians": self.model.xyz.shape[0],
        })

        # Save experiment results
        self._save_experiment(total_wall_s)

        return self._get_summary(total_wall_s)

    def _reconfigure_optimizer(self):
        """Reinitialize optimizer with updated parameter shapes (discards old state)."""
        self.optimizer = torch.optim.Adam([
            {"params": [self.model.xyz], "lr": self.config.lr_xyz * self.spatial_lr_scale,
             "eps": self.config.adam_eps, "betas": (self.config.adam_beta1, self.config.adam_beta2)},
            {"params": [self.model.rotations], "lr": self.config.lr_rotation,
             "eps": self.config.adam_eps, "betas": (self.config.adam_beta1, self.config.adam_beta2)},
            {"params": [self.model.scales], "lr": self.config.lr_scaling,
             "eps": self.config.adam_eps, "betas": (self.config.adam_beta1, self.config.adam_beta2)},
            {"params": [self.model.opacity], "lr": self.config.lr_opacity,
             "eps": self.config.adam_eps, "betas": (self.config.adam_beta1, self.config.adam_beta2)},
            {"params": [self.model.shs], "lr": self.config.lr_sh,
             "eps": self.config.adam_eps, "betas": (self.config.adam_beta1, self.config.adam_beta2)},
        ])
        torch.cuda.empty_cache()

    def _save_checkpoint(self, iteration: int, extra_metrics: Dict):
        extra_metrics["spatial_lr_scale"] = self.spatial_lr_scale
        extra_metrics["best_psnr"] = self._best_psnr
        self.checkpointer.save(
            iteration=iteration,
            model_state=self.model.get_checkpoint_state(),
            optimizer_state=self.optimizer.state_dict(),
            metrics=extra_metrics,
            label=self.config.experiment_label,
        )
        print(f"    [Saved checkpoint at iteration {iteration}]")

    def _save_experiment(self, total_wall_s: float):
        """Save complete experiment results as JSON."""
        import hashlib

        experiment = {
            "experiment_id": f"{self.config.experiment_label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "date": datetime.now(timezone.utc).isoformat(),
            "config": {
                "scene": self.config.scene,
                "resolution": self.config.resolution,
                "num_iterations": self.config.num_iterations,
                "tile_size": self.config.tile_size,
                "packed": self.config.packed,
                "sh_degree": self.config.max_sh_degree,
                "radius_clip": self.config.radius_clip,
                "eps2d": self.config.eps2d,
                "lambda_dssim": self.config.lambda_dssim,
                "lr_xyz": self.config.lr_xyz,
                "lr_rotation": self.config.lr_rotation,
                "lr_scaling": self.config.lr_scaling,
                "lr_opacity": self.config.lr_opacity,
                "lr_sh": self.config.lr_sh,
                "densification_start": self.config.densification_start,
                "densification_end": self.config.densification_end,
                "densification_grad_threshold": self.config.densification_grad_threshold,
                "prune_opacity_threshold": self.config.prune_opacity_threshold,
                "sh_degree_interval": self.config.sh_degree_interval,
                "seed": self.config.seed,
            },
            "gpu": torch.cuda.get_device_name(0),
            "total_wall_seconds": total_wall_s,
            "total_wall_minutes": total_wall_s / 60,
            "iterations_per_second": self.config.num_iterations / total_wall_s,
            "milestones": {
                "best_psnr": self._best_psnr,
                "final_psnr": self.metrics_log[-1]["psnr"] if self.metrics_log else 0.0,
                "final_gaussian_count": self.model.xyz.shape[0],
                "initial_gaussian_count": self.config.num_iterations if False else self.metrics_log[0]["num_gaussians"] if self.metrics_log else 0,
                "final_sh_degree": self.model.sh_degree,
            },
            "metrics_log": self.metrics_log,
            "psnr_trajectory": self.psnr_log,
            "loss_trajectory": self.loss_log,
            "gaussian_count_trajectory": self.gaussian_log,
        }

        # Fix initial gaussian count
        if self.metrics_log:
            experiment["milestones"]["initial_gaussian_count"] = self.metrics_log[0]["num_gaussians"]

        output_path = self.save_dir / f"{self.config.experiment_label}_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=2, default=str)
        print(f"  [Saved experiment results: {output_path}]")

    def _get_summary(self, total_wall_s: float) -> Dict:
        return {
            "experiment_label": self.config.experiment_label,
            "scene": self.config.scene,
            "tile_size": self.config.tile_size,
            "num_iterations": self.config.num_iterations,
            "total_wall_s": total_wall_s,
            "best_psnr": self._best_psnr,
            "final_gaussian_count": self.model.xyz.shape[0],
            "initial_gaussian_count": self.metrics_log[0]["num_gaussians"] if self.metrics_log else 0,
        }
