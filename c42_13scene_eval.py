#!/usr/bin/env python3
"""
C42 13-Scene Unified Evaluation Script.

Evaluates all 26 final checkpoints (13 scenes × 2 methods) with ONE identical pipeline:
  PSNR, SSIM, LPIPS, N_gaussians

Uses baseline/reference_v1/gaussian_model.py (canonical).
Evaluates on ALL cameras for each scene.
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

REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_V1_DIR = REPO_ROOT / "baseline" / "reference_v1"
sys.path.insert(0, str(REFERENCE_V1_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gaussian_model import GaussianModel
from gsplat import rasterization

sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
from dataset import GTDataset


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()


def render(model, cam, sh_degree):
    data = {
        "xyz": model.get_xyz, "rotations": model.get_rotation,
        "scales": model.get_scaling, "opacity": model.get_opacity,
        "shs": model.get_features,
    }
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    return r[0].clamp(0, 1)


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def evaluate_checkpoint(ckpt_path, scene, lpips_fn, ssim_fn):
    """Load checkpoint and evaluate PSNR/SSIM/LPIPS on ALL cameras."""
    print(f"\n  Evaluating: {Path(ckpt_path).name} (scene={scene})", flush=True)

    dataset = GTDataset(scene=scene, repo_root=str(REPO_ROOT), resolution="1080p",
                       device="cuda", background="black")
    n_cams = len(dataset)
    print(f"    {n_cams} cameras", flush=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    model.restore(ckpt, config={
        "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
        "feature_lr": 0.0025, "opacity_lr": 0.025, "scaling_lr": 0.005,
        "rotation_lr": 0.001, "percent_dense": 0.01,
    })

    psnrs, ssims, lpips_vals = [], [], []
    for ci in range(n_cams):
        cam, gt = dataset.get_item(ci)
        with torch.no_grad():
            img = render(model, cam, model.active_sh_degree)
        psnrs.append(compute_psnr(img, gt))
        ssims.append(float(1.0 - ssim_fn(img, gt).item()))
        if lpips_fn is not None:
            pred_lp = img.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            lpips_vals.append(float(lpips_fn(pred_lp, gt_lp).item()))
        if ci % 50 == 0:
            print(f"    cam {ci}/{n_cams}...", flush=True)

    result = {
        "scene": scene, "checkpoint": str(ckpt_path),
        "n_gaussians": int(model._xyz.shape[0]), "n_cameras": n_cams,
        "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    
    lpips_str = f" LPIPS={result.get('lpips','N/A')}"
    if isinstance(result.get('lpips'), float):
        lpips_str = f" LPIPS={result['lpips']:.4f}"
    print(f"    PSNR={result['psnr']:.4f} SSIM={result['ssim']:.4f}{lpips_str} N={result['n_gaussians']:,}", flush=True)

    del model
    torch.cuda.empty_cache()
    return result


# Existing 3-scene checkpoint paths
EXISTING_CKPTS = {
    "room": {
        "reference": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_30000.pt",
        "c42": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/c42_30k/checkpoints/iter_30000.pt",
    },
    "garden": {
        "reference": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/garden/checkpoints/baseline_iter_30000.pt",
        "c42": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/garden/checkpoints/c42_iter_30000.pt",
    },
    "bicycle": {
        "reference": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/bicycle/checkpoints/baseline_iter_30000.pt",
        "c42": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/bicycle/checkpoints/c42_iter_30000.pt",
    },
}

# New 10-scene checkpoint paths (trained by c42_13scene_train.py)
NEW_SCENES = [
    ("mipnerf360", "flowers"), ("mipnerf360", "stump"), ("mipnerf360", "treehill"),
    ("mipnerf360", "counter"), ("mipnerf360", "kitchen"), ("mipnerf360", "bonsai"),
    ("tanksandtemples", "truck"), ("tanksandtemples", "train"),
    ("deepblending", "drjohnson"), ("deepblending", "playroom"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scene", type=str, default=None, help="Evaluate only this scene")
    parser.add_argument("--method", type=str, default=None, help="Evaluate only this method")
    parser.add_argument("--output", type=str, default="results/c42_13scene/evaluation_results.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Init LPIPS
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
        lpips_fn.eval()
        print("LPIPS initialized (vgg)")
    except Exception as e:
        print(f"WARNING: LPIPS not available: {e}")

    ssim_fn = SepSSIM(device="cuda")

    results = {}

    # Evaluate existing 3 scenes
    for scene in ["room", "garden", "bicycle"]:
        if args.scene and args.scene != scene:
            continue
        for method in ["reference", "c42"]:
            if args.method and args.method != method:
                continue
            ckpt_path = EXISTING_CKPTS[scene][method]
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: {scene}/{method} - checkpoint not found: {ckpt_path}")
                continue
            key = f"{scene}_{method}"
            results[key] = evaluate_checkpoint(ckpt_path, scene, lpips_fn, ssim_fn)

    # Evaluate new 10 scenes
    for dataset, scene in NEW_SCENES:
        if args.scene and args.scene != scene:
            continue
        for method in ["reference", "c42"]:
            if args.method and args.method != method:
                continue
            ckpt_path = os.path.join(str(REPO_ROOT), "results/c42_13scene", dataset, scene, method,
                                    "checkpoints/iter_30000.pt")
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: {scene}/{method} - checkpoint not found: {ckpt_path}")
                continue
            key = f"{scene}_{method}"
            results[key] = evaluate_checkpoint(ckpt_path, scene, lpips_fn, ssim_fn)

    # Save results
    output_path = os.path.join(str(REPO_ROOT), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Evaluated {len(results)} checkpoints")


if __name__ == "__main__":
    main()
