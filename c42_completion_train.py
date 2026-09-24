#!/usr/bin/env python3
"""
C42 Completion Batch — Canonical C42 Training for Missing Scales.

Exact same semantics as c42_p2_training_validation_30k.py / S2.2 / S2.3:
  - Loss: (1-lambda)*L1 + lambda*d_ssim_downsampled(scale)
  - d_ssim_downsampled: F.interpolate(scale_factor, mode="area") then SepSSIM
  - SSIM: window=11, sigma=1.5, C1=(0.01)^2, C2=(0.03)^2, padding=5
  - lambda_dssim = 0.2
  - Seed = 42
  - Densification: start=500, end=15000, interval=100, grad_threshold=0.0002
  - Pruning: interval=100, opacity_threshold=0.005, reset_interval=3000
  - SH degree: interval=1000, max=3
  - Optimizer: Adam with canonical learning rates

ADDITIONS over canonical:
  - Checkpoints at 5K/10K/15K/20K/25K/30K WITH optimizer state
  - LPIPS evaluation (lpips library, net=vgg)
  - GPU selection via CUDA_VISIBLE_DEVICES
  - Parameterized scene/scale/output

Usage:
  python c42_completion_train.py --scene bicycle --scale 0.75 --gpu 0 --output results/c42_adaptive/completion_batch/bicycle_075.json
"""
import argparse
import json
import math
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Path setup (same as canonical)
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

# === EXACT CANONICAL C42 CONSTANTS (from c42_p2_training_validation_30k.py) ===
TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2
SEED = 42
NUM_ITERS = 30000
PSNR_INTERVAL = 500
SSIM_INTERVAL = 1000
LOSS_INTERVAL = 100
CHECKPOINT_INTERVAL = 5000  # CHANGED: 5K instead of 10K
DENSIFICATION_INTERVAL = 100
DENSIFICATION_GRAD_THRESHOLD = 0.0002
DENSIFICATION_START = 500
DENSIFICATION_END = 15000
CLONE_MAX_SCREEN_SIZE = 100.0
SPLIT_MAX_SCREEN_SIZE = 100.0
PRUNE_INTERVAL = 100
PRUNE_OPACITY_THRESHOLD = 0.005
PRUNE_START = 500
RESET_OPACITY_INTERVAL = 3000
SH_DEGREE_INTERVAL = 1000
MAX_SH_DEGREE = 3
EVAL_CAMERAS = list(range(0, 311, 25))  # 13 cameras (same as canonical) - will be clamped to dataset size
DENSIFICATION_PHASE_END = 15000


def get_eval_cameras(n_total):
    """Get eval camera indices, clamped to dataset size."""
    return [c for c in EVAL_CAMERAS if c < n_total]


# === EXACT CANONICAL LOSS FUNCTIONS (copied verbatim) ===

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


def d_ssim_downsampled(pred, target, scale=0.75, window_size=11, sigma=1.5, data_range=1.0):
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
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


def compute_loss(pred, gt, scale):
    l1 = F.l1_loss(pred, gt)
    if scale >= 1.0:
        dsim = d_ssim_loss(pred, gt)
    else:
        dsim = d_ssim_downsampled(pred, gt, scale=scale)
    total = (1.0 - LAMBDA_DSSIM) * l1 + LAMBDA_DSSIM * dsim
    return total, l1, dsim


def make_optimizer(model, spatial_lr_scale):
    return torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    ])


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


def evaluate(model, dataset, cam_indices, lpips_fn=None):
    psnrs, ssims, lpips_vals = [], [], []
    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        with torch.no_grad():
            pred = render(model, cam)
            mse = float(((pred - gt) ** 2).mean())
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
            psnrs.append(psnr)
            ssim_val = 1.0 - float(d_ssim_loss(pred, gt))
            ssims.append(ssim_val)
            if lpips_fn is not None:
                # LPIPS expects [B, C, H, W] in [-1, 1]
                pred_lp = pred.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
                gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
                lp = float(lpips_fn(pred_lp, gt_lp).item())
                lpips_vals.append(lp)
    result = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    return result


def get_provenance(repo_root):
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
        prov["gpu_uuid"] = props.name
        prov["gpu_sms"] = props.multi_processor_count
    except:
        prov["gpu_name"] = "unknown"

    # Trainer file hash
    trainer_path = Path(__file__)
    prov["trainer_hash"] = hashlib.sha256(trainer_path.read_bytes()).hexdigest()[:16]
    prov["trainer_path"] = str(trainer_path.name)

    try:
        import lpips
        prov["lpips_version"] = lpips.__version__
    except:
        prov["lpips_version"] = "unknown"

    return prov


def train_variant(dataset, sfm_data, scale, scene, gpu_id, output_path, save_dir):
    DEVICE = "cuda"
    print(f"\n{'='*60}")
    print(f"  C42 COMPLETION BATCH: scene={scene}, scale={scale}, GPU={gpu_id}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # LPIPS
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_fn.eval()
        print(f"  LPIPS initialized (vgg)")
    except Exception as e:
        print(f"  WARNING: LPIPS not available: {e}")
        lpips_fn = None

    model = GaussianModel(
        num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE,
    )
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=DEVICE)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"),
    )
    spatial_lr_scale = float(sfm_data["xyz"].norm(dim=-1).max().item())
    print(f"  SfM points: {model.xyz.shape[0]:,}")
    print(f"  Scene extent: {spatial_lr_scale:.2f}")

    optimizer = make_optimizer(model, spatial_lr_scale)
    num_cameras = len(dataset)

    results = {
        "scene": scene,
        "scale": scale,
        "seed": SEED,
        "num_iters": NUM_ITERS,
        "lambda_dssim": LAMBDA_DSSIM,
        "per_iter": [],
        "eval_points": [],
        "topology_events": [],
        "checkpoints": [],
        "final_eval": None,
    }

    cumulative_cloned = 0
    cumulative_split = 0
    cumulative_pruned = 0
    total_eval_time_s = 0.0
    phase_dens_iter_times = []
    phase_post_iter_times = []
    iter_times = []

    # Initial eval
    print(f"  [eval@0] ", end="", flush=True)
    eval_cams = get_eval_cameras(num_cameras)
    t_eval0 = time.perf_counter()
    eval_result = evaluate(model, dataset, eval_cams, lpips_fn)
    total_eval_time_s += time.perf_counter() - t_eval0
    ng = model.xyz.shape[0]
    results["eval_points"].append({"iter": 0, **eval_result, "n_gaussians": ng})
    print(f"PSNR={eval_result['psnr']:.2f}  SSIM={eval_result['ssim']:.4f}  GS={ng:,}")

    t_start_total = time.perf_counter()

    for iteration in range(1, NUM_ITERS + 1):
        cam_idx = (iteration - 1) % num_cameras
        cam = dataset.get_camera(cam_idx)
        gt = dataset.get_gt_image(cam_idx)

        new_degree = min(MAX_SH_DEGREE, iteration // SH_DEGREE_INTERVAL)
        if new_degree != model.sh_degree and new_degree <= MAX_SH_DEGREE:
            model.set_sh_degree(new_degree)
            optimizer = make_optimizer(model, spatial_lr_scale)

        t0 = time.perf_counter()

        pred = render(model, cam)
        loss, l1_val, dsim_val = compute_loss(pred, gt, scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Densification
        denf_count = {"cloned": 0, "split": 0, "removed": 0}
        if (iteration >= DENSIFICATION_START and iteration < DENSIFICATION_END
                and iteration % DENSIFICATION_INTERVAL == 0):
            denf_count = model.densification(
                grad_threshold=DENSIFICATION_GRAD_THRESHOLD,
                clone_max_screen_size=CLONE_MAX_SCREEN_SIZE,
                split_max_screen_size=SPLIT_MAX_SCREEN_SIZE,
            )

        # Pruning
        prune_count = 0
        if iteration >= PRUNE_START and iteration % PRUNE_INTERVAL == 0:
            prune_count = model.prune_and_reset(
                opacity_threshold=PRUNE_OPACITY_THRESHOLD,
                reset_interval=RESET_OPACITY_INTERVAL,
                current_step=iteration,
            )

        # Track topology
        if denf_count["cloned"] + denf_count["split"] + prune_count > 0:
            cumulative_cloned += denf_count["cloned"]
            cumulative_split += denf_count["split"]
            cumulative_pruned += prune_count
            optimizer = make_optimizer(model, spatial_lr_scale)
            results["topology_events"].append({
                "iter": iteration,
                "cloned": denf_count["cloned"],
                "split": denf_count["split"],
                "pruned": prune_count,
                "n_gaussians_after": model.xyz.shape[0],
                "cumulative_cloned": cumulative_cloned,
                "cumulative_split": cumulative_split,
                "cumulative_pruned": cumulative_pruned,
            })

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000
        iter_times.append(iter_ms)
        if iteration <= DENSIFICATION_PHASE_END:
            phase_dens_iter_times.append(iter_ms)
        else:
            phase_post_iter_times.append(iter_ms)

        # Loss logging
        if iteration % LOSS_INTERVAL == 0:
            results["per_iter"].append({
                "iter": iteration,
                "loss": float(loss.item()),
                "l1": float(l1_val.item()),
                "d_ssim": float(dsim_val.item()),
                "iter_ms": iter_ms,
                "n_gaussians": model.xyz.shape[0],
            })

        # PSNR/SSIM/LPIPS eval
        if iteration % PSNR_INTERVAL == 0:
            t_eval = time.perf_counter()
            eval_result = evaluate(model, dataset, eval_cams, lpips_fn)
            total_eval_time_s += time.perf_counter() - t_eval
            ng = model.xyz.shape[0]
            mean_iter_ms = float(np.mean(iter_times[-PSNR_INTERVAL:]))
            results["eval_points"].append({
                "iter": iteration, **eval_result,
                "n_gaussians": ng, "mean_iter_ms": mean_iter_ms,
            })
            lpips_str = f"  LPIPS={eval_result.get('lpips', 0):.4f}" if "lpips" in eval_result else ""
            print(f"  [eval@{iteration}] PSNR={eval_result['psnr']:.2f}  SSIM={eval_result['ssim']:.4f}{lpips_str}  "
                  f"GS={ng:,}  iter={mean_iter_ms:.1f}ms  "
                  f"clone={cumulative_cloned:,} split={cumulative_split:,} prune={cumulative_pruned:,}")

        # Checkpoint saving WITH OPTIMIZER STATE
        if iteration % CHECKPOINT_INTERVAL == 0:
            ckpt = model.get_checkpoint_state()
            # ADD optimizer state
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
            ckpt["iteration"] = iteration
            ckpt["scale"] = scale
            ckpt["scene"] = scene
            ckpt_path = save_dir / f"{scene}_s{scale}_iter_{iteration}.pt"
            torch.save(ckpt, ckpt_path)
            results["checkpoints"].append({
                "iter": iteration,
                "path": str(ckpt_path),
                "n_gaussians": model.xyz.shape[0],
                "has_optimizer_state": True,
            })
            print(f"  [ckpt@{iteration}] saved to {ckpt_path.name}  GS={model.xyz.shape[0]:,}  (with optimizer state)")

    total_time = time.perf_counter() - t_start_total

    # Final eval (all cameras)
    n_eval_cams = len(dataset)
    print(f"  [final eval, {n_eval_cams} cams] ", end="", flush=True)
    t_eval = time.perf_counter()
    all_cams = list(range(n_eval_cams))
    final_eval = evaluate(model, dataset, all_cams, lpips_fn)
    total_eval_time_s += time.perf_counter() - t_eval
    ng = model.xyz.shape[0]
    results["final_eval"] = {
        "iter": NUM_ITERS, **final_eval,
        "n_gaussians": ng, "n_eval_cameras": n_eval_cams,
    }
    lpips_str = f"  LPIPS={final_eval.get('lpips', 0):.4f}" if "lpips" in final_eval else ""
    print(f"PSNR={final_eval['psnr']:.2f}  SSIM={final_eval['ssim']:.4f}{lpips_str}  GS={ng:,}")

    # Timing
    train_time_s = total_time - total_eval_time_s
    results["timing"] = {
        "total_wall_s": total_time,
        "total_wall_min": total_time / 60,
        "train_only_s": train_time_s,
        "eval_overhead_s": total_eval_time_s,
        "mean_iter_ms": float(np.mean(iter_times)),
        "std_iter_ms": float(np.std(iter_times)),
        "min_iter_ms": float(np.min(iter_times)),
        "max_iter_ms": float(np.max(iter_times)),
        "n_iters": NUM_ITERS,
        # Steady-state (post-densification) timing
        "steady_state_mean_iter_ms": float(np.mean(phase_post_iter_times)) if phase_post_iter_times else 0,
        "steady_state_std_iter_ms": float(np.std(phase_post_iter_times)) if phase_post_iter_times else 0,
    }
    results["phase_timing"] = {
        "densification_phase": {
            "iter_range": f"1-{DENSIFICATION_PHASE_END}",
            "n_iters": DENSIFICATION_PHASE_END,
            "mean_iter_ms": float(np.mean(phase_dens_iter_times)) if phase_dens_iter_times else 0,
            "total_s": float(np.sum(phase_dens_iter_times)) / 1000,
        },
        "post_densification_phase": {
            "iter_range": f"{DENSIFICATION_PHASE_END+1}-{NUM_ITERS}",
            "n_iters": NUM_ITERS - DENSIFICATION_PHASE_END,
            "mean_iter_ms": float(np.mean(phase_post_iter_times)) if phase_post_iter_times else 0,
            "total_s": float(np.sum(phase_post_iter_times)) / 1000,
        },
    }
    results["cumulative_topology"] = {
        "total_cloned": cumulative_cloned,
        "total_split": cumulative_split,
        "total_pruned": cumulative_pruned,
    }

    print(f"\n  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Steady-state iter: {results['timing']['steady_state_mean_iter_ms']:.2f} ms")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="C42 Completion Batch Training")
    parser.add_argument("--scene", required=True, choices=["bicycle", "garden", "room"])
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Set GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 72)
    print(f"C42 Completion Batch: scene={args.scene}, scale={args.scale}, GPU={args.gpu}")
    print("=" * 72)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {gpu_name}")
    print(f"  SMs: {gpu_props.multi_processor_count}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    import gsplat
    print(f"  gsplat: {gsplat.__version__}")
    print(f"  Seed: {SEED}, Iters: {NUM_ITERS}")

    # Output dir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint save dir
    save_dir = output_path.parent / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Provenance
    prov = get_provenance(REPO_ROOT)
    prov["scene"] = args.scene
    prov["scale"] = args.scale
    prov["gpu_id_requested"] = args.gpu
    prov["dataset_resolution"] = "1080p"
    prov["loss_definition"] = "(1-lambda)*L1 + lambda*d_ssim_downsampled(scale), lambda=0.2"
    prov["scale_factor_method"] = "F.interpolate(mode='area')"
    prov["ssim_params"] = "window=11, sigma=1.5, C1=(0.01)^2, C2=(0.03)^2"
    prov["optimizer"] = "Adam, lr: xyz=1.6e-4*spatial_lr_scale, rot=1e-3, scale=5e-3, opacity=5e-2, sh=2.5e-3"
    prov["densification"] = f"start={DENSIFICATION_START}, end={DENSIFICATION_END}, interval={DENSIFICATION_INTERVAL}, grad_threshold={DENSIFICATION_GRAD_THRESHOLD}"
    prov["checkpoint_format"] = "Gaussian state + optimizer_state_dict + iteration + scale + scene"

    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene=args.scene, repo_root=REPO_ROOT, resolution="1080p", device="cuda")
    print(f"  Cameras: {len(dataset)}")

    print("\n  [Loading SfM checkpoint...]")
    sfm_data = load_initial_checkpoint(args.scene, REPO_ROOT, device="cuda")
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    # Train
    results = train_variant(dataset, sfm_data, args.scale, args.scene, args.gpu, output_path, save_dir)

    # Save
    output = {
        "experiment": "C42 Completion Batch",
        "scene": args.scene,
        "scale": args.scale,
        "provenance": prov,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
