#!/usr/bin/env python3
"""
C42 13-Scene Final Benchmark Training Script.

Trains one scene with either Reference V1 (scale=1.0) or C42 (scale=0.5).
Uses baseline/reference_v1/gaussian_model.py (canonical REFERENCE_V1_ABSGRAD).

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 c42_13scene_train.py \
    --scene flowers --scale 1.0 --gpu 0 \
    --output_dir results/c42_13scene/mipnerf360/flowers/reference
  CUDA_VISIBLE_DEVICES=1 python3 c42_13scene_train.py \
    --scene flowers --scale 0.5 --gpu 1 \
    --output_dir results/c42_13scene/mipnerf360/flowers/c42
"""
import argparse
import json
import math
import os
import sys
import time
import random
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# === PATH SETUP — reference_v1 FIRST so our GaussianModel takes priority ===
REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_V1_DIR = REPO_ROOT / "baseline" / "reference_v1"
sys.path.insert(0, str(REFERENCE_V1_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization
from colmap_reader import read_points3D_binary, sfm_to_pcd_data

sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
from dataset import GTDataset


# === SepSSIM (identical to baseline/reference_v1/trainer.py) ===
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


def d_ssim_downsampled(ssim_fn, pred, target, scale=0.5):
    """C42: downsample pred/target with area interpolation, then SSIM."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
    stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
    b = F.conv2d(stacked, ssim_fn.k_h, padding=(0, ssim_fn.padding), groups=15)
    b = F.conv2d(b, ssim_fn.k_v, padding=(ssim_fn.padding, 0), groups=15)
    mu_p, mu_t = b[:, 0:3], b[:, 3:6]
    bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
    ssim_map = (2 * mu_pt + ssim_fn.C1) * (2 * spt + ssim_fn.C2) / \
               ((mu_p2 + mu_t2 + ssim_fn.C1) * (sp2 + st2 + ssim_fn.C2))
    return 1.0 - ssim_map.mean()


def render_with_meta(model, cam, sh_degree):
    data = {
        "xyz": model.get_xyz, "rotations": model.get_rotation,
        "scales": model.get_scaling, "opacity": model.get_opacity,
        "shs": model.get_features,
    }
    r, _, meta = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    means2d_full = meta["means2d"]
    if means2d_full.requires_grad:
        means2d_full.retain_grad()
    return r[0].clamp(0, 1), meta, means2d_full


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def evaluate_model(model, dataset, ssim_fn, sh_degree, cam_indices, lpips_fn=None):
    psnrs, ssims, l1s, lpips_vals = [], [], [], []
    for ci in cam_indices:
        cam, gt = dataset.get_item(ci)
        with torch.no_grad():
            img, _, _ = render_with_meta(model, cam, sh_degree)
        psnrs.append(compute_psnr(img, gt))
        ssims.append(float(1.0 - ssim_fn(img, gt).item()))
        l1s.append(float(F.l1_loss(img, gt).item()))
        if lpips_fn is not None:
            pred_lp = img.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            lpips_vals.append(float(lpips_fn(pred_lp, gt_lp).item()))
    result = {
        "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
        "l1": float(np.mean(l1s)), "n_cameras": len(cam_indices),
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    return result


def compute_scene_extent(dataset, n_samples=50):
    centers = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        cam = dataset.get_camera(i)
        centers.append(cam.camera_center.cpu().numpy())
    centers = np.array(centers)
    extent = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            extent = max(extent, d)
    return max(extent, 0.1)


def get_provenance(repo_root, scene, scale, method):
    prov = {}
    try:
        prov["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        prov["git_describe"] = subprocess.check_output(["git", "describe", "--tags"], cwd=str(repo_root), text=True).strip()
        prov["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True).strip())
    except:
        prov["git_head"] = "unknown"
        prov["git_describe"] = "unknown"
        prov["git_dirty"] = None

    prov["pytorch_version"] = torch.__version__
    prov["cuda_version"] = torch.version.cuda
    try:
        import gsplat
        prov["gsplat_version"] = gsplat.__version__
    except:
        prov["gsplat_version"] = "unknown"

    prov["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    prov["gpu_uuid"] = torch.cuda.get_device_properties(0).name
    prov["gpu_sms"] = props.multi_processor_count

    model_path = REFERENCE_V1_DIR / "gaussian_model.py"
    config_path = REFERENCE_V1_DIR / "config.py"
    prov["gaussian_model_hash"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    prov["config_hash"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    prov["trainer_hash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prov["gaussian_model_path"] = str(model_path.relative_to(repo_root))
    prov["config_path"] = str(config_path.relative_to(repo_root))

    try:
        import lpips
        prov["lpips_version"] = lpips.__version__
    except:
        prov["lpips_version"] = "unknown"

    prov["scene"] = scene
    prov["scale"] = scale
    prov["method"] = method
    prov["iterations"] = 30000
    prov["seed"] = 42
    prov["dataset_resolution"] = "1080p"
    prov["loss_definition"] = f"(1-lambda)*L1_fullres + lambda*d_ssim_downsampled(scale={scale}), lambda=0.2"
    prov["scale_factor_method"] = "F.interpolate(mode='area')"
    prov["ssim_implementation"] = "SepSSIM window=11 sigma=1.5"
    prov["semantic_label"] = f"REFERENCE_V1 + C42 (DS-SSIM {scale})" if scale < 1.0 else "REFERENCE_V1"
    prov["config_base"] = "REFERENCE_V1_ABSGRAD"
    prov["densify_grad_threshold"] = 0.0008
    prov["lambda_dssim"] = 0.2
    prov["absgrad"] = True
    prov["grow_grad2d"] = 0.0008

    return prov


def train(scene, scale, output_dir, gpu_id, method="reference"):
    """Run training for one scene/method."""
    print(f"\n{'='*60}")
    print(f"  C42 13-SCENE BENCHMARK")
    print(f"  Scene: {scene}, Method: {method}, Scale: {scale}, GPU: {gpu_id}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # === Config ===
    config = ReferenceV1Config()
    config.scene = scene
    config.repo_root = str(REPO_ROOT)

    # === Seed ===
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # === Dataset ===
    print(f"Loading dataset: {scene}")
    dataset = GTDataset(scene=scene, repo_root=REPO_ROOT, resolution="1080p",
                       device="cuda", background="black")
    n_cameras = len(dataset)
    print(f"  {n_cameras} cameras loaded")

    # === Scene extent ===
    scene_extent = compute_scene_extent(dataset)
    print(f"  Scene extent: {scene_extent:.4f}")

    # === Model initialization ===
    print("Loading SfM point cloud...")
    sfm_path = os.path.join(str(REPO_ROOT), "data", "datasets", "mipnerf360", scene, "sparse", "0", "points3D.bin")
    if os.path.exists(sfm_path):
        print(f"  Reading COLMAP SfM points from {sfm_path}")
        points3d = read_points3D_binary(sfm_path)
        pcd_data = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)
        print(f"  SfM points: {points3d['num_points']}")
    else:
        print(f"  WARNING: SfM file not found, falling back to trained checkpoint")
        from dataset import load_initial_checkpoint
        pcd_data = load_initial_checkpoint(scene, str(REPO_ROOT), device="cuda")

    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
    initial_N = model._xyz.shape[0]

    # === Training setup ===
    model.training_setup({
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })

    # === LPIPS ===
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
        lpips_fn.eval()
        print(f"  LPIPS initialized (vgg)")
    except Exception as e:
        print(f"  WARNING: LPIPS not available: {e}")

    # === SSIM ===
    ssim_fn = SepSSIM(device="cuda")

    # === Camera sequence (frozen, same as trainer.py) ===
    viewpoint_stack = list(range(n_cameras))
    camera_sequence = []
    rng = random.Random(config.seed)
    for iteration in range(1, config.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        rand_idx = rng.randint(0, len(viewpoint_stack) - 1)
        cam_idx = viewpoint_stack.pop(rand_idx)
        camera_sequence.append(cam_idx)
    camera_sequence = np.array(camera_sequence, dtype=np.int32)

    # Save camera sequence
    np.save(os.path.join(output_dir, "camera_sequence.npy"), camera_sequence)

    # === Eval cameras (subset during training, all at end) ===
    eval_indices = list(range(0, n_cameras, max(1, n_cameras // 10)))

    # === Results ===
    results = {
        "scene": scene, "scale": scale, "method": method,
        "seed": config.seed, "num_iters": config.iterations,
        "lambda_dssim": config.lambda_dssim,
        "initial_N": initial_N, "scene_extent": float(scene_extent),
        "training_metrics": {"checkpoints": {}},
        "checkpoints_saved": [],
        "final_eval": None,
        "timing": {},
    }

    total_clones = 0
    total_splits = 0
    total_prunes = 0
    iter_times = []
    peak_vram = 0.0

    eval_iterations = set(config.eval_iterations)
    checkpoint_iterations = {5000, 10000, 15000, 20000, 25000, 30000}

    print(f"\nStarting training: {config.iterations} iterations")
    print(f"  Initial N: {initial_N}")
    print(f"  C42 SSIM scale: {scale}")

    t_start = time.perf_counter()

    for iteration in range(1, config.iterations + 1):
        model.update_learning_rate(iteration)

        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()

        cam_idx = camera_sequence[iteration - 1]
        cam, gt_image = dataset.get_item(cam_idx)

        # === Forward ===
        t0 = time.perf_counter()
        image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree)

        # === Loss ===
        L1 = F.l1_loss(image, gt_image)
        if scale >= 1.0:
            dssim = ssim_fn(image, gt_image)
        else:
            dssim = d_ssim_downsampled(ssim_fn, image, gt_image, scale=scale)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

        # === Backward ===
        loss.backward()

        # === Densification stats ===
        radii = meta["radii"][0]
        visibility_filter = (radii > 0).any(dim=-1)

        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values
            )
            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width, height=cam.image_height)

        # === Densification and pruning ===
        if (iteration > config.densify_from_iter and
            iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values
            event_result = model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii,
            )
            total_clones += event_result["cloned"]
            total_splits += event_result["split"]
            total_prunes += event_result["pruned_total"]

        # === Opacity reset ===
        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()

        # === Optimizer step ===
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000
        iter_times.append(iter_ms)

        # Track peak VRAM
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        if vram > peak_vram:
            peak_vram = vram

        # === Evaluation during training ===
        if iteration in eval_iterations or iteration == config.iterations:
            eval_result = evaluate_model(model, dataset, ssim_fn, model.active_sh_degree,
                                        eval_indices, lpips_fn)
            eval_result["n_gaussians"] = model._xyz.shape[0]
            eval_result["mean_iter_ms"] = float(np.mean(iter_times[-500:])) if iter_times else 0
            results["training_metrics"]["checkpoints"][str(iteration)] = eval_result
            lpips_str = f" LPIPS={eval_result.get('lpips','?')}"
            if isinstance(eval_result.get('lpips'), float):
                lpips_str = f" LPIPS={eval_result['lpips']:.4f}"
            print(f"  [iter {iteration:>6d}] PSNR={eval_result['psnr']:.2f} SSIM={eval_result['ssim']:.4f}"
                  f"{lpips_str} GS={model._xyz.shape[0]:,} iter={eval_result['mean_iter_ms']:.1f}ms")

        # === Checkpoint saving ===
        if iteration in checkpoint_iterations:
            ckpt = model.capture()
            ckpt["optimizer_state_dict"] = model.optimizer.state_dict()
            ckpt["iteration"] = iteration
            ckpt["scale"] = scale
            ckpt["scene"] = scene
            ckpt["method"] = method
            ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration}.pt")
            torch.save(ckpt, ckpt_path)
            results["checkpoints_saved"].append({
                "iter": iteration, "path": ckpt_path,
                "n_gaussians": model._xyz.shape[0],
            })
            if iteration % 5000 == 0:
                print(f"  [ckpt@{iteration}] saved GS={model._xyz.shape[0]:,}")

    total_time = time.perf_counter() - t_start

    # === Final eval (ALL cameras) ===
    print(f"  [final eval, all {n_cameras} cams] ", end="", flush=True)
    all_cams = list(range(n_cameras))
    final_eval = evaluate_model(model, dataset, ssim_fn, model.active_sh_degree, all_cams, lpips_fn)
    final_eval["n_gaussians"] = model._xyz.shape[0]
    final_eval["n_eval_cameras"] = n_cameras
    results["final_eval"] = final_eval
    lpips_str = f" LPIPS={final_eval.get('lpips','?')}"
    if isinstance(final_eval.get('lpips'), float):
        lpips_str = f" LPIPS={final_eval['lpips']:.4f}"
    print(f"PSNR={final_eval['psnr']:.2f} SSIM={final_eval['ssim']:.4f}{lpips_str} GS={final_eval['n_gaussians']:,}")

    # === Timing ===
    results["timing"] = {
        "total_wall_s": total_time,
        "total_wall_min": total_time / 60,
        "mean_iter_ms": float(np.mean(iter_times)),
        "std_iter_ms": float(np.std(iter_times)),
        "n_iters": config.iterations,
        "steady_state_mean_iter_ms": float(np.mean(iter_times[15000:])) if len(iter_times) > 15000 else 0,
    }
    results["total_clones"] = total_clones
    results["total_splits"] = total_splits
    results["total_prunes"] = total_prunes
    results["final_N"] = model._xyz.shape[0]
    results["peak_vram_gb"] = peak_vram
    results["provenance"] = get_provenance(REPO_ROOT, scene, scale, method)

    print(f"\n  Total wall: {total_time/60:.1f} min")
    print(f"  Mean iter: {np.mean(iter_times):.2f} ms")
    print(f"  Steady state: {results['timing']['steady_state_mean_iter_ms']:.2f} ms")
    print(f"  Final N: {model._xyz.shape[0]:,}")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")

    # === Cleanup: delete intermediate checkpoints, keep only final ===
    for ckpt_file in os.listdir(ckpt_dir):
        if ckpt_file.endswith(".pt") and ckpt_file != "iter_30000.pt":
            os.remove(os.path.join(ckpt_dir, ckpt_file))
            print(f"  [cleanup] removed {ckpt_file}")

    # === Save results ===
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(results["provenance"], f, indent=2)

    print(f"\n  Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="reference", choices=["reference", "c42"])
    args = parser.parse_args()

    train(args.scene, args.scale, args.output_dir, args.gpu, args.method)
