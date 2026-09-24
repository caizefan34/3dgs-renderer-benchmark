#!/usr/bin/env python3
"""
C42 Completion Batch — Canonical C42 Training using REFERENCE_V1 codebase.

Uses baseline/reference_v1/gaussian_model.py (official Graphdeco semantics):
  - Persistent Adam optimizer with state migration across topology changes
  - View-space mean2D gradient for densification (absgrad=True)
  - densify_grad_threshold=0.0008, max_screen_size=20, percent_dense=0.01
  - opacity_lr=0.025, packed=False
  - Fixed SH tensor with active_sh_degree progression
  - Scheduled position learning rate

C42 modification:
  - SSIM computed on F.interpolate(scale_factor=s, mode="area") downsampled pred/target
  - L1 stays at full resolution
  - loss = (1-lambda)*L1 + lambda*d_ssim_downsampled(scale)
  - lambda = 0.2

Usage:
  CUDA_VISIBLE_DEVICES=0 python c42_completion_train_v2.py --scene bicycle --scale 0.75 --output results/c42_adaptive/completion_batch/bicycle_075.json
"""
import argparse
import json
import math
import os
import sys
import time
import random
import hashlib
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

# Import reference_v1 GaussianModel BEFORE adding old scripts path
from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization
from colmap_reader import read_points3D_binary, sfm_to_pcd_data

# NOW add old scripts path for dataset.py
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
from dataset import GTDataset


# === C42 SSIM FUNCTIONS (copied from canonical, using SepSSIM pattern) ===

class SepSSIM:
    """Separable SSIM — same as baseline/reference_v1/trainer.py."""
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


def d_ssim_downsampled(ssim_fn, pred, target, scale=0.75):
    """C42 canonical: downsample pred and target with area interpolation, then SSIM."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
    # Now use the SepSSIM directly on downsampleed [B, C, H, W] tensors
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
    """Render and return (image, meta) with means2d retaining grad. Same as trainer.py."""
    data = {
        "xyz": model.get_xyz,
        "rotations": model.get_rotation,
        "scales": model.get_scaling,
        "opacity": model.get_opacity,
        "shs": model.get_features,
    }
    r, _, meta = rasterization(
        means=data["xyz"],
        quats=data["rotations"],
        scales=data["scales"],
        opacities=data["opacity"],
        colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0),
        width=cam.image_width,
        height=cam.image_height,
        tile_size=16,
        packed=False,
        sh_degree=sh_degree,
        radius_clip=0.0,
        eps2d=0.1,
        render_mode="RGB",
        absgrad=True,
    )
    means2d_full = meta["means2d"]
    if means2d_full.requires_grad:
        means2d_full.retain_grad()
    return r[0].clamp(0, 1), meta, means2d_full


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def evaluate_model(model, dataset, ssim_fn, sh_degree, cam_indices, lpips_fn=None):
    """Evaluate PSNR/SSIM/LPIPS on specified cameras."""
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
            lp = float(lpips_fn(pred_lp, gt_lp).item())
            lpips_vals.append(lp)
    result = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "l1": float(np.mean(l1s)),
        "n_cameras": len(cam_indices),
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    return result


def compute_scene_extent(dataset, n_samples=50):
    """Compute camera extent (max distance between camera centers)."""
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


def get_provenance(repo_root, scene, scale):
    """Collect provenance info."""
    import subprocess
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

    try:
        prov["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        prov["gpu_sms"] = props.multi_processor_count
    except:
        prov["gpu_name"] = "unknown"

    # Model/trainer file hashes
    model_path = REFERENCE_V1_DIR / "gaussian_model.py"
    config_path = REFERENCE_V1_DIR / "config.py"
    trainer_path = Path(__file__)
    prov["gaussian_model_hash"] = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    prov["config_hash"] = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    prov["trainer_hash"] = hashlib.sha256(trainer_path.read_bytes()).hexdigest()[:16]
    prov["gaussian_model_path"] = str(model_path.relative_to(repo_root))
    prov["config_path"] = str(config_path.relative_to(repo_root))
    prov["trainer_path"] = str(trainer_path.relative_to(repo_root))

    try:
        import lpips
        prov["lpips_version"] = lpips.__version__
    except:
        prov["lpips_version"] = "unknown"

    prov["scene"] = scene
    prov["scale"] = scale
    prov["dataset_resolution"] = "1080p"
    prov["loss_definition"] = f"(1-lambda)*L1 + lambda*d_ssim_downsampled(scale={scale}), lambda=0.2"
    prov["scale_factor_method"] = "F.interpolate(mode='area')"
    prov["ssim_implementation"] = "SepSSIM window=11 sigma=1.5 C1=(0.01)^2 C2=(0.03)^2"
    prov["semantic_label"] = f"REFERENCE_V1 + C42 (DS-SSIM {scale})"
    prov["config_base"] = "REFERENCE_V1_ABSGRAD"

    return prov


def train(scene, scale, output_path, save_dir, gpu_id):
    """Run C42 training with REFERENCE_V1 semantics."""
    print(f"\n{'='*60}")
    print(f"  C42 COMPLETION BATCH V2: scene={scene}, scale={scale}, GPU={gpu_id}")
    print(f"  Config: REFERENCE_V1 + C42 (DS-SSIM {scale})")
    print(f"{'='*60}")

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
    dataset = GTDataset(scene=scene, repo_root=REPO_ROOT, resolution="1080p", device="cuda", background="black")
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

    # === Training setup (persistent optimizer) ===
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
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
        lpips_fn.eval()
        print(f"  LPIPS initialized (vgg)")
    except Exception as e:
        print(f"  WARNING: LPIPS not available: {e}")
        lpips_fn = None

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

    # === Eval cameras ===
    eval_indices = list(range(0, n_cameras, max(1, n_cameras // 10)))  # ~10 cameras

    # === Results ===
    results = {
        "scene": scene,
        "scale": scale,
        "seed": config.seed,
        "num_iters": config.iterations,
        "lambda_dssim": config.lambda_dssim,
        "config": f"REFERENCE_V1 + C42 (DS-SSIM {scale})",
        "initial_N": initial_N,
        "scene_extent": float(scene_extent),
        "training_metrics": {"checkpoints": {}},
        "topology_events": [],
        "checkpoints_saved": [],
        "final_eval": None,
        "timing": {},
    }

    total_clones = 0
    total_splits = 0
    total_prunes = 0
    iter_times = []

    # Eval checkpoints (same as trainer.py)
    eval_iterations = set(config.eval_iterations)
    checkpoint_iterations = {5000, 10000, 15000, 20000, 25000, 30000}

    print(f"\nStarting training: {config.iterations} iterations")
    print(f"  Initial N: {initial_N}")
    print(f"  Densify: {config.densify_from_iter}-{config.densify_until_iter}, every {config.densification_interval}")
    print(f"  Grad threshold: {config.densify_grad_threshold}")
    print(f"  C42 SSIM scale: {scale}")

    t_start = time.perf_counter()

    for iteration in range(1, config.iterations + 1):
        # Update LR
        model.update_learning_rate(iteration)

        # SH progression
        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()

        # Pick camera from frozen sequence
        cam_idx = camera_sequence[iteration - 1]
        cam, gt_image = dataset.get_item(cam_idx)

        # === Forward ===
        t0 = time.perf_counter()
        image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree)

        # === Loss (C42: L1 full-res, SSIM downsampled) ===
        L1 = F.l1_loss(image, gt_image)
        if scale >= 1.0:
            dssim = ssim_fn(image, gt_image)
        else:
            dssim = d_ssim_downsampled(ssim_fn, image, gt_image, scale=scale)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

        # === Backward ===
        loss.backward()

        # === Densification stats (view-space gradient) ===
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
            N_before = model._xyz.shape[0]
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values
            event_result = model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii,
            )
            N_after = model._xyz.shape[0]
            total_clones += event_result["cloned"]
            total_splits += event_result["split"]
            total_prunes += event_result["pruned_total"]
            results["topology_events"].append({
                "iteration": iteration,
                "N_before": N_before,
                "N_after": N_after,
                **event_result,
            })

        # === Opacity reset ===
        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()

        # === Optimizer step ===
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000
        iter_times.append(iter_ms)

        # === Evaluation ===
        if iteration in eval_iterations or iteration == config.iterations:
            eval_result = evaluate_model(model, dataset, ssim_fn, model.active_sh_degree, eval_indices, lpips_fn)
            eval_result["n_gaussians"] = model._xyz.shape[0]
            eval_result["mean_iter_ms"] = float(np.mean(iter_times[-500:])) if iter_times else 0
            results["training_metrics"]["checkpoints"][str(iteration)] = eval_result
            lpips_str = f" LPIPS={eval_result.get('lpips','?')}"
            if isinstance(eval_result.get('lpips'), float):
                lpips_str = f" LPIPS={eval_result['lpips']:.4f}"
            print(f"  [iter {iteration:>6d}] PSNR={eval_result['psnr']:.2f} SSIM={eval_result['ssim']:.4f}{lpips_str} "
                  f"GS={model._xyz.shape[0]:,} iter={eval_result['mean_iter_ms']:.1f}ms")

        # === Checkpoint saving (with optimizer state) ===
        if iteration in checkpoint_iterations:
            ckpt = model.capture()
            ckpt["optimizer_state_dict"] = model.optimizer.state_dict()
            ckpt["iteration"] = iteration
            ckpt["scale"] = scale
            ckpt["scene"] = scene
            ckpt_path = save_dir / f"{scene}_s{scale}_iter_{iteration}.pt"
            torch.save(ckpt, ckpt_path)
            results["checkpoints_saved"].append({
                "iter": iteration,
                "path": str(ckpt_path),
                "n_gaussians": model._xyz.shape[0],
                "has_optimizer_state": True,
            })
            print(f"  [ckpt@{iteration}] saved {ckpt_path.name} GS={model._xyz.shape[0]:,}")

    total_time = time.perf_counter() - t_start

    # === Final eval (all cameras) ===
    print(f"  [final eval, all {n_cameras} cams] ", end="", flush=True)
    all_cams = list(range(n_cameras))
    final_eval = evaluate_model(model, dataset, ssim_fn, model.active_sh_degree, all_cams, lpips_fn)
    final_eval["n_gaussians"] = model._xyz.shape[0]
    final_eval["n_eval_cameras"] = n_cameras
    results["final_eval"] = final_eval
    lpips_str = f" LPIPS={final_eval.get('lpips','?')}"
    if isinstance(final_eval.get('lpips'), float):
        lpips_str = f" LPIPS={final_eval['lpips']:.4f}"
    print(f"PSNR={final_eval['psnr']:.2f} SSIM={final_eval['ssim']:.4f}{lpips_str} GS={model._xyz.shape[0]:,}")

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

    print(f"\n  Total wall: {total_time/60:.1f} min")
    print(f"  Mean iter: {np.mean(iter_times):.2f} ms")
    print(f"  Final N: {model._xyz.shape[0]:,}")
    print(f"  Clones: {total_clones:,} Splits: {total_splits:,} Prunes: {total_prunes:,}")

    # === Save ===
    output = {
        "experiment": "C42 Completion Batch V2",
        "scene": scene,
        "scale": scale,
        "config": f"REFERENCE_V1 + C42 (DS-SSIM {scale})",
        "provenance": get_provenance(REPO_ROOT, scene, scale),
        "results": results,
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="C42 Completion Batch Training V2 (REFERENCE_V1)")
    parser.add_argument("--scene", required=True, choices=["bicycle", "garden", "room"])
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 72)
    print(f"C42 Completion Batch V2: scene={args.scene}, scale={args.scale}, GPU={args.gpu}")
    print(f"Using: baseline/reference_v1/gaussian_model.py (official Graphdeco semantics)")
    print("=" * 72)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {gpu_name}")
    print(f"  SMs: {gpu_props.multi_processor_count}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    import gsplat
    print(f"  gsplat: {gsplat.__version__}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dir = output_path.parent / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    train(args.scene, args.scale, str(output_path), save_dir, args.gpu)


if __name__ == "__main__":
    main()
