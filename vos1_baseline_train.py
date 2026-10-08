#!/usr/bin/env python3
"""
VOS1 Baseline: Standard 3DGS training (no visibility-aware scheduling)
This is the reference for speedup and quality comparisons.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scene import Scene, GaussianModel
from utils.general_utils import seed_everything
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

def train_baseline(dataset, iteration_count=30000, seed=42):
    """Standard 3DGS training without visibility-aware scheduling."""
    seed_everything(seed)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    optimizer = torch.optim.Adam(
        gaussians.get_optimizer_params(dataset.position_lr_init, dataset.position_lr_final),
        lr=0.0, eps=1e-15
    )
    
    metrics = {
        "wall_time": 0.0,
        "optimizer_time": 0.0,
        "iteration_times": [],
        "total_updates": 0,
    }
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_cameras = scene.getTrainCameras()
    viewpoint_stack = train_cameras * (iteration_count // len(train_cameras) + 1)
    
    start_time = time.time()
    optimizer_start = time.time()
    
    for iteration in tqdm(range(1, iteration_count + 1)):
        iter_start = time.time()
        
        viewpoint = viewpoint_stack[iteration - 1]
        render_pkg = render(viewpoint, gaussians, dataset, background)
        image = render_pkg["render"]
        
        gt_image = viewpoint.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - dataset.lambda_dssim) * Ll1 + dataset.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        update_start = time.time()
        optimizer.step()
        metrics["optimizer_time"] += time.time() - update_start
        
        metrics["total_updates"] += gaussians.get_xyz.shape[0]
        
        if iteration % 1000 == 0:
            print(f"Iter {iteration}: loss={loss.item():.4f}")
        
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
        
        psnrs.append(psnr(image, gt_image).mean().item())
        ssims.append(ssim(image, gt_image).mean().item())
        
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').cuda()
        lpipss.append(lpips_fn(image, gt_image).mean().item())
    
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=".")
    args = parser.parse_args()
    
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
    
    print(f"Training {args.scene} baseline for {args.iterations} iterations")
    
    gaussians, metrics, scene = train_baseline(dataset, args.iterations, args.seed)
    eval_metrics = evaluate(gaussians, scene, dataset)
    
    results = {
        "scene": args.scene,
        "k": 1,
        "iterations": args.iterations,
        "seed": args.seed,
        "metrics": metrics,
        "quality": eval_metrics,
    }
    
    output_path = Path(args.output) / f"vos1_baseline_{Path(args.scene).name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Wall time: {metrics['wall_time']:.2f}s")
    print(f"Optimizer time: {metrics['optimizer_time']:.2f}s")
    print(f"PSNR: {eval_metrics['psnr']:.2f}dB")
    print(f"SSIM: {eval_metrics['ssim']:.4f}")
    print(f"LPIPS: {eval_metrics['lpips']:.4f}")


if __name__ == "__main__":
    main()
