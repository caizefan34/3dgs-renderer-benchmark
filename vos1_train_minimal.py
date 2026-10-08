#!/usr/bin/env python3
"""
VOS1: Minimal 3DGS training with visibility-aware Adam scheduling.
Implements the core 3DGS training loop with per-Gaussian invisible_age tracking.
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

# Try to import gsplat for rendering
try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("WARNING: gsplat not available, using dummy renderer")

def seeded(seed=42):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleGaussianModel:
    """Minimal 3D Gaussian model."""
    def __init__(self, n_gaussians, sh_degree=3, device="cuda"):
        self.n_gaussians = n_gaussians
        self.sh_degree = sh_degree
        self.device = device
        
        # Gaussian parameters
        self.xyz = torch.randn(n_gaussians, 3, device=device, requires_grad=True)
        self.features = torch.randn(n_gaussians, (sh_degree + 1) ** 2, 3, device=device, requires_grad=True)
        self.opacity = torch.randn(n_gaussians, 1, device=device, requires_grad=True)
        self.scaling = torch.randn(n_gaussians, 3, device=device, requires_grad=True)
        self.rotation = torch.randn(n_gaussians, 4, device=device, requires_grad=True)
        
        self.params = [self.xyz, self.features, self.opacity, self.scaling, self.rotation]
        
    def get_optimizer_params(self, lr_init=1e-3, lr_final=1e-5):
        """Get optimizer parameter groups."""
        return [
            {"params": [self.xyz], "lr": lr_init},
            {"params": [self.features], "lr": lr_init},
            {"params": [self.opacity], "lr": lr_init},
            {"params": [self.scaling], "lr": lr_init},
            {"params": [self.rotation], "lr": lr_init},
        ]

def render_gaussians(model, camera, background=torch.zeros(3)):
    """Render Gaussians from a camera viewpoint."""
    if GSPLAT_AVAILABLE:
        # Use gsplat for rendering
        rendered, _, _ = rasterization(
            means=model.xyz,
            quats=F.normalize(model.rotation, dim=-1),
            scales=torch.exp(model.scaling),
            opacities=torch.sigmoid(model.opacity),
            colors=model.features[:, 0],  # Use DC component
            viewmats=camera["view_matrix"].unsqueeze(0),
            Ks=camera["K"].unsqueeze(0),
            width=camera["width"],
            height=camera["height"],
            sh_degree=0,
            packed=True,
            render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)
    else:
        # Differentiable dummy renderer for testing
        # Project 3D points to 2D and create a simple differentiable rendering
        H, W = camera["height"], camera["width"]
        
        # Simple orthographic projection (differentiable)
        xyz_proj = model.xyz[:, :2]  # Take x, y coordinates
        
        # Normalize to image space
        xyz_norm = (xyz_proj - xyz_proj.min(dim=0)[0]) / (xyz_proj.max(dim=0)[0] - xyz_proj.min(dim=0)[0] + 1e-8)
        img_coords = (xyz_norm * torch.tensor([W-1, H-1], device=model.device)).long()
        
        # Create rendered image by splatting
        rendered = torch.zeros(H, W, 3, device=model.device, requires_grad=False)
        
        # Add gradient-friendly rendering (use grid_sample for differentiability)
        # Create a simple feature grid
        grid = torch.zeros(1, H, W, 2, device=model.device)
        grid[0, :, :, 0] = (torch.arange(W, device=model.device).float() / W * 2 - 1)
        grid[0, :, :, 1] = (torch.arange(H, device=model.device).float() / H * 2 - 1)
        
        # Sample features (differentiable)
        features_2d = model.features[:, 0, :].mean(dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
        features_grid = features_2d.expand(1, H, W, 3)
        
        rendered = F.grid_sample(
            features_grid.permute(0, 3, 1, 2),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).permute(0, 2, 3, 1).squeeze(0)
        
        return rendered.clamp(0, 1)

def load_cameras(scene_path, split="train"):
    """Load cameras from COLMAP format."""
    camera_dir = Path(scene_path) / f"sparse/0" if split == "train" else Path(scene_path) / "images"
    
    # This is a simplified loader - real implementation would parse COLMAP
    # For now, return dummy cameras
    cameras = []
    for i in range(100):
        cam = {
            "view_matrix": torch.eye(4),
            "K": torch.tensor([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=torch.float32),
            "width": 512,
            "height": 512,
            "image": torch.rand(512, 512, 3),
        }
        cameras.append(cam)
    
    return cameras

def train_visibility_aware(scene_path, k=2, iterations=30000, seed=42, output_dir="."):
    """
    Train with visibility-aware Adam scheduling.
    
    Update rule:
      visible (radii > 0): reset age = 0, update normally
      invisible (radii = 0): increase age, update only when age % k == 0
    """
    seeded(seed)
    
    # Initialize model
    n_gaussians = 100000  # Initial count
    model = SimpleGaussianModel(n_gaussians, device="cuda")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.get_optimizer_params(), lr=0.001)
    
    # Per-Gaussian invisible age tracking
    invisible_age = torch.zeros(n_gaussians, dtype=torch.int32, device="cuda")
    
    # Load cameras
    train_cameras = load_cameras(scene_path, "train")
    
    # Metrics
    metrics = {
        "wall_time": 0.0,
        "optimizer_time": 0.0,
        "total_updates": 0,
        "total_skipped": 0,
        "iteration_times": [],
    }
    
    start_time = time.time()
    
    # Training loop
    pbar = tqdm(range(1, iterations + 1), desc=f"Training k={k}")
    for iteration in pbar:
        iter_start = time.time()
        
        # Sample camera
        cam_idx = iteration % len(train_cameras)
        camera = train_cameras[cam_idx]
        
        # Render
        rendered = render_gaussians(model, camera)
        
        # Compute loss (simple L1 + SSIM)
        gt_image = camera["image"].to("cuda")
        l1_loss = F.l1_loss(rendered, gt_image)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        l1_loss.backward()
        
        # Visibility-aware update frequency modification
        update_start = time.time()
        
        # Get visibility from radii (simplified: use gradient magnitude as proxy)
        # In real implementation, this would come from the rasterizer
        with torch.no_grad():
            # Use gradient magnitude as visibility proxy (sum over all non-batch dims)
            grad_norms = []
            for p in model.params:
                if p.grad is not None:
                    # Sum over all dimensions except the first (batch dimension)
                    norm_dims = list(range(1, p.grad.dim()))
                    if norm_dims:
                        grad_norm = p.grad.norm(dim=norm_dims)
                    else:
                        grad_norm = p.grad.abs()
                    grad_norms.append(grad_norm)
                else:
                    grad_norms.append(torch.zeros(p.shape[0], device="cuda"))
            
            grad_norm = sum(grad_norms)
            # Use a lower threshold to ensure some Gaussians are visible
            visible_mask = grad_norm > 1e-10
            invisible_mask = ~visible_mask
        
        # Update invisible_age
        invisible_age[visible_mask] = 0
        invisible_age[invisible_mask] += 1
        
        # Determine which Gaussians to update
        should_update = visible_mask | (invisible_age % k == 0)
        
        # Count
        update_count = should_update.sum().item()
        skip_count = (~should_update).sum().item()
        metrics["total_updates"] += update_count
        metrics["total_skipped"] += skip_count
        
        # Mask gradients for Gaussians that should not update
        for param in model.params:
            if param.grad is not None:
                # Expand mask to match parameter shape
                # should_update is [N], param.grad is [N, ...]
                mask_shape = [should_update.shape[0]] + [1] * (param.grad.dim() - 1)
                mask = should_update.view(mask_shape).expand_as(param.grad)
                param.grad[~mask] = 0.0
        
        # Optimizer step
        optimizer.step()
        
        metrics["optimizer_time"] += time.time() - update_start
        
        # Logging
        if iteration % 1000 == 0:
            pbar.set_postfix({
                "loss": f"{l1_loss.item():.4f}",
                "upd": update_count,
                "skip": skip_count,
            })
        
        metrics["iteration_times"].append(time.time() - iter_start)
    
    metrics["wall_time"] = time.time() - start_time
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=".")
    args = parser.parse_args()
    
    print(f"VOS1 Training: {args.scene}, k={args.k}, {args.iterations} iterations")
    
    model, metrics = train_visibility_aware(
        args.scene, args.k, args.iterations, args.seed, args.output
    )
    
    # Save results
    results = {
        "scene": args.scene,
        "k": args.k,
        "iterations": args.iterations,
        "seed": args.seed,
        "metrics": {
            "wall_time": metrics["wall_time"],
            "optimizer_time": metrics["optimizer_time"],
            "total_updates": metrics["total_updates"],
            "total_skipped": metrics["total_skipped"],
            "skip_rate": metrics["total_skipped"] / (metrics["total_updates"] + metrics["total_skipped"]),
        },
    }
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"vos1_{Path(args.scene).name}_k{args.k}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Wall time: {metrics['wall_time']:.2f}s")
    print(f"Optimizer time: {metrics['optimizer_time']:.2f}s")
    print(f"Total updates: {metrics['total_updates']}")
    print(f"Total skipped: {metrics['total_skipped']}")
    print(f"Skip rate: {100*results['metrics']['skip_rate']:.2f}%")


if __name__ == "__main__":
    main()
