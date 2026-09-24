#!/usr/bin/env python3
"""B1 vs B1A deterministic short training — semantic freeze test.

Runs 3DGS training for 800 iterations with densification (starting at 500),
using identical seed, init, camera sequence, optimizer, and densification config.

Usage:
  python3 short_train.py --variant B1 --gsplat-path INSTALLED --accutile false --out /tmp/results/b1_train.json
  python3 short_train.py --variant B1A --gsplat-path /path/to/gsplat-true-accutile --accutile true --out /tmp/results/b1a_train.json
"""
import sys, os, json, argparse, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass

# Add repo paths
REPO_ROOT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo"
sys.path.insert(0, f"{REPO_ROOT}/src")
sys.path.insert(0, REPO_ROOT)

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TrainConfig:
    num_iterations: int = 800
    tile_size: int = 16
    packed: bool = True
    sh_degree: int = 0  # starts at 0, increases every 1000 iters
    max_sh_degree: int = 3
    sh_degree_interval: int = 1000
    lambda_dssim: float = 0.2
    lr_xyz: float = 1.6e-4
    lr_rotation: float = 1e-3
    lr_scaling: float = 5e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3
    adam_eps: float = 1e-15
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    spatial_lr_scale: float = 1.0
    densification_interval: int = 100
    densification_grad_threshold: float = 0.0002
    densification_start: int = 500
    densification_end: int = 15000
    clone_max_screen_size: float = 100.0
    split_max_screen_size: float = 100.0
    prune_interval: int = 100
    prune_opacity_threshold: float = 0.005
    prune_start: int = 500
    reset_opacity_interval: int = 3000
    grad_clip: float = 1.0

# ── Loss ──

def d_ssim_loss(pred, target, window_size=11, sigma=1.5):
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    elif pred.ndim == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
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

def combined_loss(pred, target, lambda_dssim=0.2):
    l1 = F.l1_loss(pred, target)
    dsim = d_ssim_loss(pred, target)
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * dsim
    return total, l1, dsim

# ── Gaussian Model ──

class GaussianModel:
    def __init__(self, xyz, rotations, scales_log, opacity_logit, shs, device="cuda"):
        self.xyz = nn.Parameter(xyz.to(device).contiguous())
        self.rotations = nn.Parameter(rotations.to(device).contiguous())
        self.scales = nn.Parameter(scales_log.to(device).contiguous())
        self.opacity = nn.Parameter(opacity_logit.to(device).contiguous())
        # Truncate SH to degree 0 (1 coefficient) for initial training
        self.shs = nn.Parameter(shs[:, :1, :].to(device).contiguous())
        self.sh_degree = 0
        self.device = device
        self._xyz_grad_accum = None
        self._denf_steps = 0

    def parameters(self):
        return [self.xyz, self.rotations, self.scales, self.opacity, self.shs]

    def forward(self):
        return {
            "xyz": self.xyz,
            "rotations": F.normalize(self.rotations, dim=-1).contiguous(),
            "scales": torch.exp(self.scales).contiguous(),
            "opacity": torch.sigmoid(self.opacity).squeeze(-1).contiguous(),
            "shs": self.shs.contiguous(),
        }

    @torch.no_grad()
    def accumulate_positional_gradient(self):
        if self.xyz.grad is None:
            return
        grad = self.xyz.grad.detach()
        grad_norm = grad.norm(dim=-1)
        if self._xyz_grad_accum is None or self._xyz_grad_accum.shape != grad_norm.shape:
            self._xyz_grad_accum = grad_norm
            self._denf_steps = 1
        else:
            self._xyz_grad_accum = self._xyz_grad_accum + grad_norm
            self._denf_steps += 1

    @torch.no_grad()
    def densification(self, grad_threshold=2e-4, clone_max_screen_size=100.0, split_max_screen_size=100.0):
        if self._xyz_grad_accum is None or self._denf_steps == 0:
            return {"cloned": 0, "split": 0, "removed": 0}
        avg_grad = self._xyz_grad_accum / self._denf_steps
        high_grad_mask = avg_grad >= grad_threshold
        self._xyz_grad_accum = None
        self._denf_steps = 0
        scales_activated = torch.exp(self.scales).detach()
        median_scale = scales_activated.median(dim=0).values
        is_small = (scales_activated <= median_scale).all(dim=-1)
        clone_mask = high_grad_mask & is_small
        split_mask = high_grad_mask & (~is_small)
        clone_count, split_count = clone_mask.sum().item(), split_mask.sum().item()
        if clone_count == 0 and split_count == 0:
            return {"cloned": 0, "split": 0, "removed": 0}
        original_xyz = self.xyz.detach()
        original_rotations = self.rotations.detach()
        original_scales = self.scales.detach()
        original_opacity = self.opacity.detach()
        original_shs = self.shs.detach()
        new_xyz_parts, new_rot_parts, new_scale_parts, new_opa_parts, new_shs_parts = [], [], [], [], []
        if clone_count > 0:
            cloned_xyz = original_xyz[clone_mask]
            noise = torch.randn_like(cloned_xyz) * 0.01 * torch.exp(original_scales[clone_mask])
            new_xyz_parts.append(cloned_xyz + noise)
            new_rot_parts.append(original_rotations[clone_mask])
            new_scale_parts.append(original_scales[clone_mask])
            new_opa_parts.append(original_opacity[clone_mask])
            new_shs_parts.append(original_shs[clone_mask])
        if split_count > 0:
            split_xyz = original_xyz[split_mask]
            split_rot = original_rotations[split_mask]
            split_scales = original_scales[split_mask] - math.log(2.0)
            split_opa = original_opacity[split_mask]
            split_shs = original_shs[split_mask]
            new_xyz_parts.append(torch.cat([split_xyz, split_xyz], dim=0))
            new_rot_parts.append(torch.cat([split_rot, split_rot], dim=0))
            new_scale_parts.append(torch.cat([split_scales, split_scales], dim=0))
            new_opa_parts.append(torch.cat([split_opa, split_opa], dim=0))
            new_shs_parts.append(torch.cat([split_shs, split_shs], dim=0))
        new_xyz = torch.cat(new_xyz_parts, dim=0)
        new_rot = torch.cat(new_rot_parts, dim=0)
        new_scales = torch.cat(new_scale_parts, dim=0)
        new_opa = torch.cat(new_opa_parts, dim=0)
        new_shs = torch.cat(new_shs_parts, dim=0)
        if split_count > 0:
            split_noise = torch.randn_like(new_xyz[-split_count * 2:]) * 0.0025
            new_xyz[-split_count * 2:] = new_xyz[-split_count * 2:] + split_noise
        self.xyz = nn.Parameter(torch.cat([self.xyz, new_xyz], dim=0).contiguous())
        self.rotations = nn.Parameter(torch.cat([self.rotations, new_rot], dim=0).contiguous())
        self.scales = nn.Parameter(torch.cat([self.scales, new_scales], dim=0).contiguous())
        self.opacity = nn.Parameter(torch.cat([self.opacity, new_opa], dim=0).contiguous())
        self.shs = nn.Parameter(torch.cat([self.shs, new_shs], dim=0).contiguous())
        self._xyz_grad_accum = None
        self._denf_steps = 0
        return {"cloned": clone_count, "split": split_count, "removed": 0}

    @torch.no_grad()
    def prune_and_reset(self, opacity_threshold=0.005, reset_interval=3000, current_step=0):
        opacities = torch.sigmoid(self.opacity).detach().squeeze(-1)
        prune_mask = opacities < opacity_threshold
        removed = prune_mask.sum().item()
        if removed > 0:
            keep_mask = ~prune_mask
            self.xyz = nn.Parameter(self.xyz[keep_mask].contiguous())
            self.rotations = nn.Parameter(self.rotations[keep_mask].contiguous())
            self.scales = nn.Parameter(self.scales[keep_mask].contiguous())
            self.opacity = nn.Parameter(self.opacity[keep_mask].contiguous())
            self.shs = nn.Parameter(self.shs[keep_mask].contiguous())
        if current_step > 0 and current_step % reset_interval == 0:
            opacities = torch.sigmoid(self.opacity).detach().squeeze(-1)
            near_threshold = (opacities < opacity_threshold * 10) & (opacities >= opacity_threshold)
            reset_count = near_threshold.sum().item()
            if reset_count > 0:
                reset_val = torch.logit(torch.full((reset_count, 1), opacity_threshold * 2, device=self.device))
                self.opacity.data[near_threshold] = reset_val
        self._xyz_grad_accum = None
        self._denf_steps = 0
        return removed

    def get_param_groups(self, config):
        return [
            {"params": [self.xyz], "lr": config.lr_xyz},
            {"params": [self.rotations], "lr": config.lr_rotation},
            {"params": [self.scales], "lr": config.lr_scaling},
            {"params": [self.opacity], "lr": config.lr_opacity},
            {"params": [self.shs], "lr": config.lr_sh},
        ]

# ── Main training ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gsplat-path", required=True)
    parser.add_argument("--accutile", type=str, default="false")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="room")
    parser.add_argument("--num-iters", type=int, default=800)
    args = parser.parse_args()

    set_seed(SEED)

    if args.gsplat_path != "INSTALLED":
        sys.path.insert(0, args.gsplat_path)
    import gsplat
    from gsplat import rasterization

    accutile = args.accutile == "true"
    device = "cuda"
    config = TrainConfig()
    config.num_iterations = args.num_iters

    # Load data
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    from PIL import Image

    scene = args.scene
    official_dir = f"{REPO_ROOT}/data/official/mipnerf360/{scene}"
    gt_dir = f"{REPO_ROOT}/data/datasets/mipnerf360/{scene}/images_4_png"

    print(f"[{args.variant}] Loading PLY...")
    sfm = load_ply(f"{official_dir}/point_cloud.ply", device="cpu")

    print(f"[{args.variant}] Loading cameras...")
    cams = load_cameras_from_json(f"{official_dir}/cameras.json", device="cpu")

    # Get target resolution from first image
    first_img = Image.open(f"{gt_dir}/DSCF4667.png")
    target_w, target_h = first_img.size
    cams = resize_cameras(cams, target_w, target_h)
    print(f"[{args.variant}] Resized cameras to {target_w}x{target_h}")

    # Build image index
    img_index = {}
    for fpath in Path(gt_dir).iterdir():
        if fpath.is_file() and fpath.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            img_index[fpath.stem] = fpath

    # Match cameras to images
    valid_cams = []
    valid_img_paths = []
    for cam in cams:
        img_name = getattr(cam, "image_name", None)
        if img_name and Path(img_name).stem in img_index:
            valid_cams.append(cam)
            valid_img_paths.append(img_index[Path(img_name).stem])

    num_cameras = len(valid_cams)
    print(f"[{args.variant}] {num_cameras} cameras matched to images")

    # Pre-cache GT images
    print(f"[{args.variant}] Caching GT images...")
    gt_images = []
    for i, path in enumerate(valid_img_paths):
        with Image.open(path) as im:
            rgba_np = np.array(im.convert("RGBA"), dtype=np.uint8)
        gt_images.append(torch.from_numpy(rgba_np).to(device, dtype=torch.float32)[..., :3].div_(255.0).contiguous())

    # Initialize model
    print(f"[{args.variant}] Initializing model with {sfm['xyz'].shape[0]} Gaussians...")
    N = sfm["xyz"].shape[0]
    opacity_logit = torch.logit(torch.full((N, 1), 0.1))
    model = GaussianModel(
        xyz=sfm["xyz"],
        rotations=sfm["rotations"],
        scales_log=sfm["scales"],
        opacity_logit=opacity_logit,
        shs=sfm["shs"],
        device=device,
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.get_param_groups(config),
        lr=0.0,  # LR is per-group
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    # Training loop
    print(f"\n[{args.variant}] Starting training for {config.num_iterations} iterations...")
    log = []
    for iteration in range(config.num_iterations):
        # Camera selection (round-robin)
        cam_idx = iteration % num_cameras
        cam = valid_cams[cam_idx]
        gt_image = gt_images[cam_idx]

        # Move camera tensors to device
        viewmat = cam.viewmatrix.to(device).unsqueeze(0)  # [1, 4, 4]
        K = cam.K.to(device).unsqueeze(0)  # [1, 3, 3]
        width = cam.image_width
        height = cam.image_height

        # Forward
        data = model.forward()
        kwargs = dict(
            sh_degree=model.sh_degree, absgrad=False,
            tile_size=config.tile_size, packed=config.packed,
            render_mode="RGB",
        )
        if accutile:
            kwargs["accutile"] = True
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=viewmat, Ks=K, width=width, height=height, **kwargs,
        )
        rendered = rendered[0].clamp(0, 1)

        # Loss
        loss, l1, dsim = combined_loss(rendered, gt_image, config.lambda_dssim)
        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()

        # Accumulate positional gradient
        model.accumulate_positional_gradient()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)

        # Optimizer step
        optimizer.step()
        torch.cuda.synchronize()

        # Densification & Pruning
        denf_count = {"cloned": 0, "split": 0, "removed": 0}
        if (iteration >= config.densification_start and
                iteration < config.densification_end and
                iteration % config.densification_interval == 0):
            denf_count = model.densification(
                grad_threshold=config.densification_grad_threshold,
                clone_max_screen_size=config.clone_max_screen_size,
                split_max_screen_size=config.split_max_screen_size,
            )

        prune_count = 0
        if (iteration >= config.prune_start and
                iteration % config.prune_interval == 0):
            prune_count = model.prune_and_reset(
                opacity_threshold=config.prune_opacity_threshold,
                reset_interval=config.reset_opacity_interval,
                current_step=iteration,
            )

        # Rebuild optimizer if topology changed
        if denf_count["cloned"] + denf_count["split"] + prune_count > 0:
            optimizer = torch.optim.Adam(
                model.get_param_groups(config),
                lr=0.0, betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
            )

        # Log
        if iteration % 10 == 0 or iteration < 20:
            entry = {
                "iter": iteration,
                "loss": float(loss.item()),
                "l1": float(l1.item()),
                "psnr": psnr,
                "n_gaussians": model.xyz.shape[0],
                "cloned": denf_count["cloned"],
                "split": denf_count["split"],
                "pruned": prune_count,
            }
            log.append(entry)
            if iteration % 100 == 0:
                print(f"  [{args.variant}] iter {iteration}: loss={loss.item():.6f} psnr={psnr:.2f} N={model.xyz.shape[0]} "
                      f"clone={denf_count['cloned']} split={denf_count['split']} prune={prune_count}")

    # Final state
    result = {
        "variant": args.variant,
        "gsplat_version": gsplat.__version__,
        "accutile": accutile,
        "seed": SEED,
        "scene": scene,
        "num_iterations": config.num_iterations,
        "image_resolution": [target_w, target_h],
        "initial_gaussians": N,
        "final_gaussians": model.xyz.shape[0],
        "log": log,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{args.variant}] Done. Final N={model.xyz.shape[0]}, saved to {args.out}")

if __name__ == "__main__":
    main()
