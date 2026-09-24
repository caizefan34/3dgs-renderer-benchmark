#!/usr/bin/env python3
"""Quick test: evaluate canonical Room baseline 30K checkpoint with our pipeline."""
import json, math, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use idle GPU 3

REPO_ROOT = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset

TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0

def d_ssim_loss(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    def blur(x):
        return F.conv2d(x, kernel, padding=window_size // 2, groups=pred.shape[1])
    mu_pred = blur(pred)
    mu_target = blur(target)
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    sigma_pred_sq = blur(pred ** 2) - mu_pred_sq
    sigma_target_sq = blur(target ** 2) - mu_target_sq
    sigma_pred_target = blur(pred * target) - mu_pred_target
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return 1.0 - ssim_map.mean()

def render(model, cam):
    data = model.forward()
    rendered, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
    )
    return rendered[0].clamp(0, 1)

# Load canonical Room baseline checkpoint
ckpt_path = REPO_ROOT / "results" / "reference_v1" / "room_30k" / "checkpoints" / "iter_30000.pt"
print(f"Loading: {ckpt_path}")
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")
print(f"num_points: {ckpt.get('num_points')}, sh_degree: {ckpt.get('sh_degree')}")

model = GaussianModel.from_checkpoint_state(ckpt, device="cuda")
print(f"Model loaded: {model.xyz.shape[0]} Gaussians, sh_degree={model.sh_degree}")

# Load dataset
dataset = GTDataset(scene="room", repo_root=REPO_ROOT, resolution="1080p", device="cuda")
print(f"Dataset: {len(dataset)} cameras")

# Evaluate on first 13 cameras (same as EVAL_CAMERAS)
eval_cams = list(range(0, min(311, len(dataset)), 25))
print(f"Evaluating on cameras: {eval_cams}")

psnrs, ssims = [], []
for ci in eval_cams:
    cam = dataset.get_camera(ci)
    gt = dataset.get_gt_image(ci)
    with torch.no_grad():
        pred = render(model, cam)
        mse = float(((pred - gt) ** 2).mean())
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
        ssim_val = 1.0 - float(d_ssim_loss(pred, gt))
        psnrs.append(psnr)
        ssims.append(ssim_val)
        print(f"  cam {ci}: PSNR={psnr:.2f} SSIM={ssim_val:.4f} pred_range=[{pred.min():.3f},{pred.max():.3f}] gt_range=[{gt.min():.3f},{gt.max():.3f}]")

print(f"\nMean PSNR={np.mean(psnrs):.4f} SSIM={np.mean(ssims):.4f}")
print(f"(Canonical reference: PSNR=32.30 SSIM=0.9263)")
