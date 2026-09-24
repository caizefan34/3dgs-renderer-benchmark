"""
R0.3 Continuation Runner — Forward-Time Gate Discovery.

Runs 200 consecutive iterations from a checkpoint, computing at EVERY iteration:
  - Visible-conditional correlations: Pearson/Spearman between each forward signal
    and G_opt among ONLY visible Gaussians
  - Visible-conditional gradient-mass coverage: TopK(forward signal | visible) → G_opt mass
  - Visibility confound decomposition: all-Gaussian vs visible-only coverage
  - Work-weighted Pareto: gradient mass retained vs work retained
  - Visible fraction per iteration

Forward signals tested:
  - tiles_per_gauss (from meta)
  - projected_radius (max of radii [N,2])
  - opacity (model.get_opacity, pre-sigmoid → sigmoid)

Does NOT save per-Gaussian arrays — only aggregate statistics per iteration.
Uses explicit gaussian_id propagation from R0.2.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 continuation_runner.py \
    --checkpoint results/reference_v1/ckpt_14k/checkpoints/iter_2000.pt \
    --start-iter 2000 --n-iters 200 \
    --output results/reference_v1/r0.3/window_2000 \
    --camera-sequence results/reference_v1/room_30k/camera_sequence.npy
"""

import sys
import os
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline", "reference_v1")
SRC_DIR = os.path.join(REPO_ROOT, "src")

sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, SRC_DIR)

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization

OLD_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "epic05", "phase7")
sys.path.insert(0, OLD_SCRIPTS)
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


def render_with_meta(model, cam, sh_degree):
    r, _, meta = rasterization(
        means=model.get_xyz, quats=model.get_rotation,
        scales=model.get_scaling, opacities=model.get_opacity,
        colors=model.get_features,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    means2d_full = meta["means2d"]
    means2d_full.retain_grad()
    return r, meta, means2d_full


def extract_gopt(model):
    grads = {}
    if model._xyz.grad is not None:
        grads["xyz_norm"] = model._xyz.grad.norm(dim=-1).detach()
    else:
        grads["xyz_norm"] = torch.zeros(model._xyz.shape[0], device="cuda")
    if model._opacity.grad is not None:
        grads["opacity_abs"] = model._opacity.grad.abs().detach()
    else:
        grads["opacity_abs"] = torch.zeros(model._opacity.shape[0], device="cuda")
    if model._scaling.grad is not None:
        grads["scale_norm"] = model._scaling.grad.norm(dim=-1).detach()
    else:
        grads["scale_norm"] = torch.zeros(model._scaling.shape[0], device="cuda")
    if model._rotation.grad is not None:
        grads["rot_norm"] = model._rotation.grad.norm(dim=-1).detach()
    else:
        grads["rot_norm"] = torch.zeros(model._rotation.shape[0], device="cuda")
    if model._shs.grad is not None:
        grads["shs_norm"] = model._shs.grad.flatten(start_dim=1).norm(dim=-1).detach()
    else:
        grads["shs_norm"] = torch.zeros(model._shs.shape[0], device="cuda")
    total_sq = (grads["xyz_norm"]**2 + grads["opacity_abs"]**2 +
                grads["scale_norm"]**2 + grads["rot_norm"]**2 + grads["shs_norm"]**2)
    grads["total_norm"] = torch.sqrt(total_sq + 1e-20)
    return {k: v.cpu().numpy().astype(np.float32) for k, v in grads.items()}


def compute_visible_conditional_correlation(signal, gopt, vis_mask):
    """Compute Pearson/Spearman between signal and gopt among visible Gaussians only."""
    s = signal[vis_mask]
    g = gopt[vis_mask]
    n = len(s)
    if n < 10:
        return {"pearson": 0, "spearman": 0, "n": n}

    # Pearson
    if np.std(s) > 1e-10 and np.std(g) > 1e-10:
        pearson = float(np.corrcoef(s, g)[0, 1])
    else:
        pearson = 0.0

    # Spearman
    rank_s = np.argsort(np.argsort(s)).astype(float)
    rank_g = np.argsort(np.argsort(g)).astype(float)
    if np.std(rank_s) > 1e-10 and np.std(rank_g) > 1e-10:
        spearman = float(np.corrcoef(rank_s, rank_g)[0, 1])
    else:
        spearman = 0.0

    return {"pearson": pearson, "spearman": spearman, "n": n}


def compute_visible_coverage(signal, gopt, vis_mask, k_pcts=(10, 20, 32, 50)):
    """Compute gradient-mass coverage: TopK(signal | visible) → G_opt mass / total visible G_opt."""
    s = signal[vis_mask]
    g = gopt[vis_mask]
    n = len(s)
    if n < 10:
        return {f"k{k}": 0 for k in k_pcts}

    total_g = g.sum()
    if total_g < 1e-20:
        return {f"k{k}": 0 for k in k_pcts}

    result = {}
    for k_pct in k_pcts:
        k = max(1, int(n * k_pct / 100))
        top_idx = np.argsort(s)[::-1][:k]  # TopK by signal
        coverage = float(g[top_idx].sum() / total_g)
        result[f"k{k_pct}"] = coverage

    # Also compute oracle (TopK by G_opt itself)
    for k_pct in k_pcts:
        k = max(1, int(n * k_pct / 100))
        oracle_idx = np.argsort(g)[::-1][:k]
        result[f"oracle_k{k_pct}"] = float(g[oracle_idx].sum() / total_g)

    # Random baseline (average of 10 trials)
    for k_pct in k_pcts:
        k = max(1, int(n * k_pct / 100))
        rand_covs = []
        for _ in range(10):
            rand_idx = np.random.choice(n, k, replace=False)
            rand_covs.append(float(g[rand_idx].sum() / total_g))
        result[f"random_k{k_pct}"] = float(np.mean(rand_covs))

    return result


def compute_all_gaussian_coverage(signal, gopt, vis_mask, k_pcts=(10, 20, 32, 50)):
    """Compute coverage over ALL Gaussians (including invisible) for confound decomposition."""
    n = len(signal)
    total_g = gopt.sum()
    if total_g < 1e-20:
        return {f"k{k}": 0 for k in k_pcts}

    result = {}
    for k_pct in k_pcts:
        k = max(1, int(n * k_pct / 100))
        top_idx = np.argsort(signal)[::-1][:k]
        coverage = float(gopt[top_idx].sum() / total_g)
        result[f"k{k_pct}"] = coverage

    return result


def compute_work_pareto(signal, gopt, tiles, vis_mask, k_pcts=(10, 20, 32, 50)):
    """Compute Pareto: gradient mass retained vs work retained for TopK(signal | visible)."""
    s = signal[vis_mask]
    g = gopt[vis_mask]
    t = tiles[vis_mask]
    n = len(s)
    if n < 10:
        return {}

    total_g = g.sum()
    total_t = t.sum()
    if total_g < 1e-20 or total_t < 1e-20:
        return {}

    result = {}
    for k_pct in k_pcts:
        k = max(1, int(n * k_pct / 100))
        top_idx = np.argsort(s)[::-1][:k]

        grad_mass_retained = float(g[top_idx].sum() / total_g)
        work_retained = float(t[top_idx].sum() / total_t)
        work_removed = 1.0 - work_retained

        result[f"k{k_pct}"] = {
            "gradient_mass_retained": grad_mass_retained,
            "work_retained": work_retained,
            "work_removed": work_removed,
        }

    return result


def run_continuation(checkpoint_path, start_iter, n_iters, output_dir,
                     config, camera_sequence_path, gpu_id=0):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"

    # Dataset
    print(f"Loading dataset: {config.scene}")
    dataset = GTDataset(config.scene, config.repo_root)
    n_cameras = len(dataset)
    centers = []
    for i in range(min(50, n_cameras)):
        cam = dataset.get_camera(i)
        centers.append(cam.camera_center.cpu().numpy())
    centers = np.array(centers)
    scene_extent = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            scene_extent = max(scene_extent, d)
    scene_extent = max(scene_extent, 0.1)
    print(f"  {n_cameras} cameras, extent={scene_extent:.4f}")

    camera_sequence = np.load(camera_sequence_path)
    print(f"  Camera sequence: {len(camera_sequence)} entries")

    # Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint N: {ckpt['num_points']}, SH degree: {ckpt['active_sh_degree']}")

    model = GaussianModel(max_sh_degree=config.sh_degree)
    training_config = {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    }
    model.restore(ckpt, training_config)
    N = model._xyz.shape[0]
    print(f"  Model restored: N={N}, SH degree={model.active_sh_degree}")

    ssim_fn = SepSSIM(device=device)

    # Data storage
    visible_fraction = {}  # iter -> float
    correlations = {}  # iter -> {signal: {pearson, spearman, n}}
    visible_coverage = {}  # iter -> {signal: {k10, k20, k32, k50, oracle_k*, random_k*}}
    all_gauss_coverage = {}  # iter -> {signal: {k10, k20, k32, k50}} (for confound decomposition)
    work_pareto = {}  # iter -> {signal: {k: {gradient_mass_retained, work_retained, work_removed}}}

    print(f"\nStarting continuation: {n_iters} iterations (from iter {start_iter+1})")

    for i in range(n_iters):
        iteration = start_iter + 1 + i
        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(iteration)

        # Forward
        image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree)
        if image.ndim == 4:
            image = image.squeeze(0)
        radii = meta["radii"][0]  # [N, 2]
        visibility_filter = (radii > 0).any(dim=-1)  # [N]

        # Loss
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

        # Backward
        loss.backward()

        # Extract gradients
        gopt = extract_gopt(model)

        # Extract forward signals (per-Gaussian, on CPU as numpy)
        vis_np = visibility_filter.cpu().numpy().astype(bool)
        tiles = meta["tiles_per_gauss"][0].float().detach().cpu().numpy().astype(np.float32)
        projected_radius = radii.float().max(dim=-1).values.detach().cpu().numpy().astype(np.float32)
        opacity = model.get_opacity.detach().cpu().numpy().astype(np.float32)

        # Visible fraction
        vis_frac = float(vis_np.sum() / len(vis_np))
        visible_fraction[iteration] = vis_frac

        # === Part 4: Visible-conditional correlations ===
        iter_corr = {}
        for signal_name, signal_vals in [
            ("tiles_per_gauss", tiles),
            ("projected_radius", projected_radius),
            ("opacity", opacity),
        ]:
            iter_corr[signal_name] = {}
            for gopt_name in ["total_norm", "xyz_norm", "opacity_abs",
                              "scale_norm", "rot_norm", "shs_norm"]:
                iter_corr[signal_name][gopt_name] = compute_visible_conditional_correlation(
                    signal_vals, gopt[gopt_name], vis_np
                )
        correlations[iteration] = iter_corr

        # === Part 5: Visible-conditional gradient-mass coverage ===
        iter_vis_cov = {}
        for signal_name, signal_vals in [
            ("tiles_per_gauss", tiles),
            ("projected_radius", projected_radius),
            ("opacity", opacity),
        ]:
            iter_vis_cov[signal_name] = compute_visible_coverage(
                signal_vals, gopt["total_norm"], vis_np
            )
        visible_coverage[iteration] = iter_vis_cov

        # === Part 6: All-Gaussian coverage (for confound decomposition) ===
        iter_all_cov = {}
        for signal_name, signal_vals in [
            ("tiles_per_gauss", tiles),
            ("projected_radius", projected_radius),
            ("opacity", opacity),
        ]:
            iter_all_cov[signal_name] = compute_all_gaussian_coverage(
                signal_vals, gopt["total_norm"], vis_np
            )
        all_gauss_coverage[iteration] = iter_all_cov

        # === Part 7+8: Work-weighted Pareto ===
        iter_pareto = {}
        for signal_name, signal_vals in [
            ("tiles_per_gauss", tiles),
            ("projected_radius", projected_radius),
            ("opacity", opacity),
        ]:
            iter_pareto[signal_name] = compute_work_pareto(
                signal_vals, gopt["total_norm"], tiles, vis_np
            )
        work_pareto[iteration] = iter_pareto

        # Densification stats
        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values
            )
            model.add_densification_stats(means2d, visibility_filter,
                                           width=cam.image_width, height=cam.image_height)

        # Topology events
        if (iteration > config.densify_from_iter and
            iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values
            model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii,
            )

        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [iter {iteration}] N={model._xyz.shape[0]} loss={loss.item():.6f} "
                  f"vis_frac={vis_frac:.3f}")

    # === Save ===
    print("\n=== Saving outputs ===")

    provenance = {
        "experiment": "r0.3_continuation",
        "semantic_label": "REFERENCE_V1_ABSGRAD",
        "checkpoint_path": checkpoint_path,
        "start_iter": start_iter,
        "n_iters": n_iters,
        "gpu_id": gpu_id,
        "scene": config.scene,
        "N_at_checkpoint": ckpt["num_points"],
        "N_at_end": model._xyz.shape[0],
        "camera_sequence": camera_sequence_path,
        "identity_tracking": "explicit_gaussian_id",
        "forward_signals_tested": ["tiles_per_gauss", "projected_radius", "opacity"],
    }
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    with open(os.path.join(output_dir, "visible_fraction.json"), "w") as f:
        json.dump({str(k): v for k, v in visible_fraction.items()}, f, indent=2)
    with open(os.path.join(output_dir, "visible_conditional_correlations.json"), "w") as f:
        json.dump({str(k): v for k, v in correlations.items()}, f, indent=2)
    with open(os.path.join(output_dir, "visible_gradient_mass_coverage.json"), "w") as f:
        json.dump({str(k): v for k, v in visible_coverage.items()}, f, indent=2)
    with open(os.path.join(output_dir, "all_gaussian_coverage.json"), "w") as f:
        json.dump({str(k): v for k, v in all_gauss_coverage.items()}, f, indent=2)
    with open(os.path.join(output_dir, "work_gradient_pareto.json"), "w") as f:
        json.dump({str(k): v for k, v in work_pareto.items()}, f, indent=2)

    print(f"  Outputs saved to: {output_dir}")
    print(f"  Final N: {model._xyz.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R0.3 Continuation Runner")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start-iter", type=int, required=True)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--camera-sequence", required=True)
    args = parser.parse_args()

    config = ReferenceV1Config(scene="room", iterations=30000)
    run_continuation(
        checkpoint_path=args.checkpoint,
        start_iter=args.start_iter,
        n_iters=args.n_iters,
        output_dir=args.output,
        config=config,
        camera_sequence_path=args.camera_sequence,
        gpu_id=args.gpu,
    )
