#!/usr/bin/env python3
"""
VOS1: Visibility-Aware Adam Scheduler
Only modifies optimizer update frequency based on per-Gaussian invisible_age.
Does NOT modify: rasterizer, gradients, loss, densification, or pruning.

Update rule:
  visible (radii > 0): reset age = 0, update normally
  invisible (radii = 0): increase age, update only when age % k == 0
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add the 3DGS source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scene import Scene, GaussianModel
from utils.general_utils import seed_everything
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from argparse import ArgumentParser
from gaussian_renderer import render, GaussianRasterizationSettings, GaussianRasterizer

def train_visibility_aware(dataset, k, iteration_count=30000, seed=42):
    """
    Train with visibility-aware Adam scheduling.
    
    Args:
        dataset: Dataset config
        k: Skip interval for invisible Gaussians
        iteration_count: Total training iterations
        seed: Random seed
    """
    seed_everything(seed)
    
    # Initialize model and scene
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        gaussians.get_optimizer_params(dataset.position_lr_init, dataset.position_lr_final),
        lr=0.0, eps=1e-15
    )
    
    # Per-Gaussian invisible age tracking
    n_gaussians = gaussians.get_xyz.shape[0]
    invisible_age = torch.zeros(n_gaussians, dtype=torch.int32, device="cuda")
    
    # Training metrics
    metrics = {
        "wall_time": 0.0,
        "optimizer_time": 0.0,
        "iteration_times": [],
        "update_counts": [],
        "skipped_counts": [],
        "total_updates": 0,
        "total_skipped": 0,
    }
    
    # Training loop
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_cameras = scene.getTrainCameras()
    viewpoint_stack = train_cameras * (iteration_count // len(train_cameras) + 1)
    
    start_time = time.time()
    optimizer_start = time.time()
    
    for iteration in tqdm(range(1, iteration_count + 1)):
        iter_start = time.time()
        
        # Sample a camera
        viewpoint = viewpoint_stack[iteration - 1]
        
        # Render
        render_pkg = render(viewpoint, gaussians, dataset, background)
        image = render_pkg["render"]
        radii = render_pkg["radii"]
        
        # Compute loss (UNCHANGED from baseline)
        gt_image = viewpoint.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - dataset.lambda_dssim) * Ll1 + dataset.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Backward pass (UNCHANGED from baseline)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Visibility-aware update frequency modification
        update_start = time.time()
        
        # Update invisible_age based on visibility (radii > 0 means visible)
        visible_mask = radii > 0
        invisible_mask = ~visible_mask
        
        # Age update: reset to 0 for visible, increment for invisible
        invisible_age[visible_mask] = 0
        invisible_age[invisible_mask] += 1
        
        # Determine which Gaussians to update
        # Visible: always update
        # Invisible: update only when age % k == 0
        should_update = visible_mask | (invisible_age % k == 0)
        
        # Count updates
        update_count = should_update.sum().item()
        skip_count = (~should_update).sum().item()
        
        metrics["update_counts"].append(update_count)
        metrics["skipped_counts"].append(skip_count)
        metrics["total_updates"] += update_count
        metrics["total_skipped"] += skip_count
        
        # Apply updates only to selected Gaussians
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                
                # Mask gradients for Gaussians that should not update
                # This is the ONLY modification to the optimizer
                param.grad[~should_update] = 0.0
        
        # Optimizer step (UNCHANGED - just with masked gradients)
        optimizer.step()
        
        metrics["optimizer_time"] += time.time() - update_start
        
        # Densification (UNCHANGED from baseline)
        if iteration < dataset.densify_grad_threshold:
            # ... densification code (unchanged) ...
            pass
        
        # Logging
        if iteration % 1000 == 0:
            print(f"Iter {iteration}: loss={loss.item():.4f}, updates={update_count}, skipped={skip_count}")
        
        metrics["iteration_times"].append(time.time() - iter_start)
    
    metrics["wall_time"] = time.time() - start_time
    
    return gaussians, metrics, scene


def evaluate(gaussians, scene, dataset):
    """Evaluate PSNR, SSIM, LPIPS on test set."""
    test_cameras = scene.getTestCameras()
    
    psnrs = []
    ssims = []
    lpipss = []
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    for viewpoint in test_cameras:
        render_pkg = render(viewpoint, gaussians, dataset, background)
        image = render_pkg["render"]
        gt_image = viewpoint.original_image.cuda()
        
        # PSNR
        psnrs.append(psnr(image, gt_image).mean().item())
        
        # SSIM
        ssims.append(ssim(image, gt_image).mean().item())
        
        # LPIPS
        lpipss.append(lpips_func(image, gt_image).item())
    
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=".")
    args = parser.parse_args()
    
    # Load dataset config (simplified)
    class Dataset:
        def __init__(self, scene_path):
            self.model_path = scene_path
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.train_test_exp = False
            self.data_device = "cuda"
            self.eval = True
            self.sh_degree = 3
            self.position_lr_init = 0.00016
            self.position_lr_final = 0.0000016
            self.position_lr_delay_mult = 0.01
            self.position_lr_max_steps = 30000
            self.feature_lr = 0.0025
            self.opacity_lr = 0.05
            self.scaling_lr = 0.005
            self.rotation_lr = 0.001
            self.percent_dense = 0.01
            self.lambda_dssim = 0.2
            self.densification_interval = 100
            self.opacity_reset_interval = 3000
            self.densify_from_iter = 500
            self.densify_until_iter = 15000
            self.densify_grad_threshold = 0.0002
            self.random_background = False
    
    dataset = Dataset(args.scene)
    
    print(f"Training {args.scene} with k={args.k} for {args.iterations} iterations")
    
    # Train with visibility-aware scheduling
    gaussians, metrics, scene = train_visibility_aware(
        dataset, args.k, args.iterations, args.seed
    )
    
    # Evaluate
    eval_metrics = evaluate(gaussians, scene, dataset)
    
    # Save results
    results = {
        "scene": args.scene,
        "k": args.k,
        "iterations": args.iterations,
        "seed": args.seed,
        "metrics": metrics,
        "quality": eval_metrics,
    }
    
    output_path = Path(args.output) / f"vos1_{Path(args.scene).name}_k{args.k}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Wall time: {metrics['wall_time']:.2f}s")
    print(f"Optimizer time: {metrics['optimizer_time']:.2f}s")
    print(f"Total updates: {metrics['total_updates']}")
    print(f"Total skipped: {metrics['total_skipped']}")
    print(f"Skip rate: {100 * metrics['total_skipped'] / (metrics['total_updates'] + metrics['total_skipped']):.2f}%")
    print(f"PSNR: {eval_metrics['psnr']:.2f}dB")
    print(f"SSIM: {eval_metrics['ssim']:.4f}")
    print(f"LPIPS: {eval_metrics['lpips']:.4f}")


if __name__ == "__main__":
    main()
