"""
R0.2 Continuation Runner — C51 Final Mechanism Gate evidence collection.

Runs 200 consecutive iterations from a checkpoint, collecting at EVERY iteration:
  - G_opt: per-Gaussian optimization gradient norms (xyz, opacity, scale, rotation, SH, total)
  - G_dens: per-Gaussian densification gradient (means2d.absgrad, pixel-space scaled)
  - Workload: tiles_per_gauss
  - Camera transition: center distance, view-direction angle

R0.2 NEW metrics (beyond R0.1):
  - Part A: Corrected Jaccard/Recall/Precision (separate, not conflated)
  - Part B: Predictive gradient-mass coverage (Oracle/Previous/Random for K=50%, 32%)
  - Part D: Per-Gaussian G_dens ↔ G_opt correlation at same iteration
  - Part H: Explicit gaussian_id propagation (no xyz-hash inference)

Historical C51 mask signal = xyz grad norm (NOT G_opt_total).
R0.2 measures BOTH xyz_norm and total_norm to enable valid historical comparison.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 continuation_runner.py \
    --checkpoint results/reference_v1/ckpt_14k/checkpoints/iter_2000.pt \
    --start-iter 2000 --n-iters 200 \
    --output results/reference_v1/r0.2/window_2000 \
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

# === Path setup ===
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


# === SSIM (exact copy from trainer.py) ===
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


# === Camera transition ===
def compute_camera_transition(cam_t, cam_t1):
    center_t = cam_t.camera_center
    center_t1 = cam_t1.camera_center
    center_dist = float(torch.norm(center_t - center_t1).item())
    view_dir_t = -cam_t.viewmatrix[:3, 2]
    view_dir_t1 = -cam_t1.viewmatrix[:3, 2]
    cos_angle = float(torch.dot(view_dir_t, view_dir_t1).item())
    cos_angle = max(-1.0, min(1.0, cos_angle))
    view_angle = math.degrees(math.acos(cos_angle))
    return center_dist, view_angle


# === Render ===
def render_with_meta(model, cam, sh_degree):
    r, _, meta = rasterization(
        means=model.get_xyz,
        quats=model.get_rotation,
        scales=model.get_scaling,
        opacities=model.get_opacity,
        colors=model.get_features,
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
    means2d_full.retain_grad()
    return r, meta, means2d_full


# === G_opt extraction ===
def extract_gopt(model):
    """Extract per-Gaussian optimization gradient norms after backward()."""
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


# === G_dens extraction ===
def extract_gdens(means2d, visibility_filter, width, height):
    grad = getattr(means2d, "absgrad", None)
    if grad is None:
        grad = means2d.grad
    if grad is None:
        N = means2d.shape[1] if means2d.dim() == 3 else means2d.shape[0]
        return np.zeros(N, dtype=np.float32), np.zeros(N, dtype=bool)
    if grad.dim() == 3:
        grad = grad[0]
    grad = grad.clone()
    grad[:, 0] *= width / 2.0
    grad[:, 1] *= height / 2.0
    gdens_norm = grad.norm(dim=-1).detach().cpu().numpy().astype(np.float32)
    gdens_vis = visibility_filter.detach().cpu().numpy().astype(bool)
    return gdens_norm, gdens_vis


# === Part A: Corrected Top-K Jaccard/Recall/Precision ===
def compute_topk_corrected(vals_t, vals_t1, ids_t, ids_t1, k_pcts=(50, 32)):
    """Compute CORRECTED Jaccard, Recall, Precision for Top-K.

    Jaccard_K = |TopK_t ∩ TopK_{t-1}| / |TopK_t ∪ TopK_{t-1}|
    Recall_K  = |TopK_t ∩ TopK_{t-1}| / |TopK_t|
    Precision_K = |TopK_t ∩ TopK_{t-1}| / |TopK_{t-1}|

    When |TopK_t| = |TopK_{t-1}| = K, Recall = Precision = |intersection|/K,
    but Jaccard = |intersection| / (2K - |intersection|) < Recall.
    """
    id_to_idx_t1 = {int(gid): i for i, gid in enumerate(ids_t1)}
    matched_t = []
    matched_t1 = []
    for i, gid in enumerate(ids_t):
        gid = int(gid)
        if gid in id_to_idx_t1:
            matched_t.append(vals_t[i])
            matched_t1.append(vals_t1[id_to_idx_t1[gid]])
    n = len(matched_t)
    if n < 10:
        return {"n_matched": n}
    arr_t = np.array(matched_t)
    arr_t1 = np.array(matched_t1)
    result = {"n_matched": n}
    for k_pct in k_pcts:
        k = max(1, int(n * k_pct / 100))
        top_t = set(np.argsort(arr_t)[::-1][:k].tolist())
        top_t1 = set(np.argsort(arr_t1)[::-1][:k].tolist())
        inter = len(top_t & top_t1)
        union = len(top_t | top_t1)
        result[f"jaccard{k_pct}"] = float(inter / union) if union > 0 else 0.0
        result[f"recall{k_pct}"] = float(inter / k)  # = |TopK_t ∩ TopK_{t-1}| / |TopK_t|
        result[f"precision{k_pct}"] = float(inter / k)  # = |TopK_t ∩ TopK_{t-1}| / |TopK_{t-1}|
    return result


# === Part B: Predictive Gradient-Mass Coverage ===
def compute_coverage(vals_t1, vals_t, ids_t1, ids_t, k_pcts=(50, 32), n_random=10):
    """Compute gradient-mass coverage for Oracle, Previous, Random predictors.

    Coverage_K(t) = sum_{i in TopK(predictor)} G_opt_i(t) / sum_i G_opt_i(t)

    Oracle: TopK from G_opt(t) itself
    Previous: TopK from G_opt(t-1), matched to t by ID
    Random: fixed-size random subset
    """
    id_to_idx_t = {int(gid): i for i, gid in enumerate(ids_t)}
    total_t = vals_t.sum()
    if total_t < 1e-20:
        return {"n_matched": 0}

    # Match t-1 Gaussians to t by ID
    matched_t1_vals = []  # vals at t-1 for matched Gaussians
    matched_t_idx = []     # indices in vals_t for matched Gaussians
    for i, gid in enumerate(ids_t1):
        gid = int(gid)
        if gid in id_to_idx_t:
            matched_t1_vals.append(vals_t1[i])
            matched_t_idx.append(id_to_idx_t[gid])
    n_matched = len(matched_t_idx)
    if n_matched < 10:
        return {"n_matched": n_matched}

    matched_t1_vals = np.array(matched_t1_vals)
    matched_t_vals = vals_t[matched_t_idx]  # G_opt(t) at matched positions

    result = {"n_matched": n_matched}
    for k_pct in k_pcts:
        k = max(1, int(n_matched * k_pct / 100))

        # Oracle: TopK from current G_opt(t)
        oracle_idx = np.argsort(matched_t_vals)[::-1][:k]
        oracle_cov = float(matched_t_vals[oracle_idx].sum() / total_t)
        result[f"oracle_k{k_pct}"] = oracle_cov

        # Previous: TopK from G_opt(t-1), then look up their G_opt(t) values
        prev_top_idx = np.argsort(matched_t1_vals)[::-1][:k]
        prev_cov = float(matched_t_vals[prev_top_idx].sum() / total_t)
        result[f"previous_k{k_pct}"] = prev_cov

        # Random: average over n_random trials
        rand_covs = []
        for _ in range(n_random):
            rand_idx = np.random.choice(n_matched, k, replace=False)
            rand_covs.append(float(matched_t_vals[rand_idx].sum() / total_t))
        result[f"random_k{k_pct}_mean"] = float(np.mean(rand_covs))
        result[f"random_k{k_pct}_std"] = float(np.std(rand_covs))

    return result


# === Part D: Per-Gaussian G_dens ↔ G_opt correlation ===
def compute_gdens_gopt_correlation(gdens_norm, gopt_total, gopt_xyz, gdens_vis):
    """Compute per-Gaussian correlation between G_dens and G_opt at same iteration.

    Only visible Gaussians have meaningful G_dens.
    """
    vis = gdens_vis
    gd = gdens_norm[vis]
    gt = gopt_total[vis]
    gx = gopt_xyz[vis]

    n = len(gd)
    if n < 10:
        return {"n": n}

    result = {"n": n}

    # G_dens vs G_opt_total
    if np.std(gd) > 1e-10 and np.std(gt) > 1e-10:
        result["pearson_gdens_gopt_total"] = float(np.corrcoef(gd, gt)[0, 1])
    else:
        result["pearson_gdens_gopt_total"] = 0.0

    rank_gd = np.argsort(np.argsort(gd)).astype(float)
    rank_gt = np.argsort(np.argsort(gt)).astype(float)
    if np.std(rank_gd) > 1e-10 and np.std(rank_gt) > 1e-10:
        result["spearman_gdens_gopt_total"] = float(np.corrcoef(rank_gd, rank_gt)[0, 1])
    else:
        result["spearman_gdens_gopt_total"] = 0.0

    # G_dens vs G_opt_xyz (historical C51 signal)
    if np.std(gd) > 1e-10 and np.std(gx) > 1e-10:
        result["pearson_gdens_gopt_xyz"] = float(np.corrcoef(gd, gx)[0, 1])
    else:
        result["pearson_gdens_gopt_xyz"] = 0.0

    rank_gx = np.argsort(np.argsort(gx)).astype(float)
    if np.std(rank_gd) > 1e-10 and np.std(rank_gx) > 1e-10:
        result["spearman_gdens_gopt_xyz"] = float(np.corrcoef(rank_gd, rank_gx)[0, 1])
    else:
        result["spearman_gdens_gopt_xyz"] = 0.0

    # Top50(G_dens) → G_opt_total mass coverage
    total_gt = gt.sum()
    if total_gt > 1e-20:
        for k_pct in [50, 32]:
            k = max(1, int(n * k_pct / 100))
            top_gdens_idx = np.argsort(gd)[::-1][:k]
            cov = float(gt[top_gdens_idx].sum() / total_gt)
            result[f"top{k_pct}_gdens_covers_gopt_total"] = cov

            top_gdens_idx_x = np.argsort(gd)[::-1][:k]
            total_gx = gx.sum()
            if total_gx > 1e-20:
                cov_x = float(gx[top_gdens_idx_x].sum() / total_gx)
                result[f"top{k_pct}_gdens_covers_gopt_xyz"] = cov_x

    # Oracle: Top50(G_opt_total) → G_opt_total mass (for reference)
    total_gt = gt.sum()
    if total_gt > 1e-20:
        for k_pct in [50, 32]:
            k = max(1, int(n * k_pct / 100))
            oracle_idx = np.argsort(gt)[::-1][:k]
            result[f"oracle_top{k_pct}_gopt_total"] = float(gt[oracle_idx].sum() / total_gt)

    return result


# === Concentration (C49, for continuity) ===
def compute_concentration(values, visible_mask=None):
    if visible_mask is not None:
        vals = values[visible_mask]
    else:
        vals = values
    n = len(vals)
    if n == 0:
        return {"gini": 0, "top10_mass": 0, "top50_mass": 0, "n": 0}
    total = vals.sum()
    if total < 1e-20:
        return {"gini": 0, "top10_mass": 0, "top50_mass": 0, "n": n}
    sorted_vals = np.sort(vals)[::-1]
    cumsum = np.cumsum(sorted_vals)
    masses = {}
    for pct in [1, 5, 10, 20, 32, 50]:
        k = max(1, int(n * pct / 100))
        masses[f"top{pct}_mass"] = float(cumsum[k-1] / total)
    sorted_all = np.sort(vals)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_all) / (n * total)) - (n + 1) / n)
    return {**masses, "gini": gini, "n": n}


# === Lag-1 metrics (for continuity with R0.1) ===
def compute_lag1_metrics(vals_t, vals_t1, ids_t, ids_t1):
    """Compute lag-1 Pearson/Spearman (for continuity). Top-K in Part A."""
    id_to_idx_t1 = {int(gid): i for i, gid in enumerate(ids_t1)}
    matched_t = []
    matched_t1 = []
    for i, gid in enumerate(ids_t):
        gid = int(gid)
        if gid in id_to_idx_t1:
            matched_t.append(vals_t[i])
            matched_t1.append(vals_t1[id_to_idx_t1[gid]])
    n = len(matched_t)
    if n < 10:
        return {"pearson": 0, "spearman": 0, "n_matched": n}
    arr_t = np.array(matched_t)
    arr_t1 = np.array(matched_t1)
    if np.std(arr_t) > 1e-10 and np.std(arr_t1) > 1e-10:
        pearson = float(np.corrcoef(arr_t, arr_t1)[0, 1])
    else:
        pearson = 0.0
    rank_t = np.argsort(np.argsort(arr_t)).astype(float)
    rank_t1 = np.argsort(np.argsort(arr_t1)).astype(float)
    if np.std(rank_t) > 1e-10 and np.std(rank_t1) > 1e-10:
        spearman = float(np.corrcoef(rank_t, rank_t1)[0, 1])
    else:
        spearman = 0.0
    return {"pearson": pearson, "spearman": spearman, "n_matched": n}


# === Main ===
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

    # Camera sequence
    camera_sequence = np.load(camera_sequence_path)
    print(f"  Camera sequence: {len(camera_sequence)} entries")

    # Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint N: {ckpt['num_points']}, SH degree: {ckpt['active_sh_degree']}")

    # Model
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
    print(f"  Explicit ID tracking: IDs 0..{N-1}, next_id={model._next_gaussian_id}")

    ssim_fn = SepSSIM(device=device)

    # Data storage
    c49_gopt = {}
    c49_gdens = {}
    c50_gopt = {}  # lag-1 pearson/spearman (continuity)
    topk_corrected = {}  # Part A: per-pair corrected Jaccard/Recall/Precision
    coverage_gopt_total = {}  # Part B: coverage for G_opt_total
    coverage_gopt_xyz = {}    # Part B: coverage for xyz_norm (historical C51 signal)
    coverage_per_param = {}   # Part B: per-parameter coverage
    gdens_gopt_corr = {}  # Part D: per-iteration G_dens↔G_opt correlation
    camera_transitions = {}
    topology_events = []
    prev_data = None

    print(f"\nStarting continuation: {n_iters} iterations (from iter {start_iter+1})")
    print(f"  Densification at iters where (iter % 100 == 0) and (500 < iter < 15000)")

    for i in range(n_iters):
        iteration = start_iter + 1 + i
        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(iteration)

        # Forward
        image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree)
        if image.ndim == 4:
            image = image.squeeze(0)
        radii = meta["radii"][0]
        visibility_filter = (radii > 0).any(dim=-1)

        # Loss
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

        # Backward
        loss.backward()

        # Extract gradients
        gopt = extract_gopt(model)
        gdens_norm, gdens_vis = extract_gdens(
            means2d, visibility_filter, cam.image_width, cam.image_height
        )
        tiles = meta["tiles_per_gauss"][0].float().detach().cpu().numpy().astype(np.float32)

        # Current IDs (explicit, from model — matches gradient data size)
        ids_for_prev = model._gaussian_ids.cpu().numpy().tolist()
        current_N = model._xyz.shape[0]

        # C49 concentration
        c49_gopt[iteration] = compute_concentration(gopt["total_norm"], gdens_vis)
        for pname in ["xyz_norm", "opacity_abs", "scale_norm", "rot_norm", "shs_norm"]:
            c49_gopt[iteration][f"_{pname}"] = compute_concentration(gopt[pname], gdens_vis)
        c49_gdens[iteration] = compute_concentration(gdens_norm, gdens_vis)

        # Part D: Per-Gaussian G_dens ↔ G_opt correlation (same iteration)
        gdens_gopt_corr[iteration] = compute_gdens_gopt_correlation(
            gdens_norm, gopt["total_norm"], gopt["xyz_norm"], gdens_vis
        )

        # Lag-1 metrics (if previous data exists)
        if prev_data is not None:
            center_dist, view_angle = compute_camera_transition(prev_data["cam"], cam)
            camera_transitions[iteration] = {
                "center_dist": center_dist,
                "view_angle": view_angle,
                "cam_t": prev_data["cam_idx"],
                "cam_t1": cam_idx,
            }

            # C50 lag-1 (continuity)
            c50_gopt[iteration] = compute_lag1_metrics(
                prev_data["gopt_total"], gopt["total_norm"],
                prev_data["ids"], ids_for_prev
            )

            # Part A: Corrected Top-K for G_opt_total
            topk_corrected[iteration] = {
                "gopt_total": compute_topk_corrected(
                    prev_data["gopt_total"], gopt["total_norm"],
                    prev_data["ids"], ids_for_prev
                ),
                "gopt_xyz": compute_topk_corrected(
                    prev_data["gopt_xyz"], gopt["xyz_norm"],
                    prev_data["ids"], ids_for_prev
                ),
            }

            # Part B: Coverage for G_opt_total
            coverage_gopt_total[iteration] = compute_coverage(
                prev_data["gopt_total"], gopt["total_norm"],
                prev_data["ids"], ids_for_prev
            )

            # Part B: Coverage for xyz_norm (historical C51 signal)
            coverage_gopt_xyz[iteration] = compute_coverage(
                prev_data["gopt_xyz"], gopt["xyz_norm"],
                prev_data["ids"], ids_for_prev
            )

            # Part B: Per-parameter coverage (K=50 only)
            param_cov = {}
            for pname in ["xyz_norm", "opacity_abs", "scale_norm", "rot_norm", "shs_norm"]:
                param_cov[pname] = compute_coverage(
                    prev_data["gopt"][pname], gopt[pname],
                    prev_data["ids"], ids_for_prev, k_pcts=(50,)
                )
            coverage_per_param[iteration] = param_cov

        # Densification stats
        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values
            )
            model.add_densification_stats(means2d, visibility_filter,
                                           width=cam.image_width, height=cam.image_height)

        # Topology events
        topology_event = "none"
        if (iteration > config.densify_from_iter and
            iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):

            N_before = model._xyz.shape[0]
            ids_before_count = model._gaussian_ids.shape[0]

            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values

            if size_threshold is not None:
                event_result = model.densify_and_prune(
                    max_grad=config.densify_grad_threshold,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    max_screen_size=size_threshold,
                    radii=current_radii,
                )
            else:
                event_result = model.densify_and_prune(
                    max_grad=config.densify_grad_threshold,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    max_screen_size=None,
                    radii=current_radii,
                )

            # IDs are now explicitly propagated by the model (Part H)
            # No xyz-hash reconciliation needed
            topology_event = "clone_split"
            topology_events.append({
                "iteration": iteration,
                "N_before": N_before,
                "N_after": model._xyz.shape[0],
                "ids_before": ids_before_count,
                "ids_after": model._gaussian_ids.shape[0],
                "next_gaussian_id": model._next_gaussian_id,
                **event_result,
            })
            current_N = model._xyz.shape[0]

        # Opacity reset
        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()
            topology_event = "prune_reset" if topology_event == "none" else topology_event

        # Optimizer step
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        # Store previous data
        # ids_for_prev matches gopt/gdens/tiles size (pre-densification)
        prev_data = {
            "gopt_total": gopt["total_norm"],
            "gopt_xyz": gopt["xyz_norm"],
            "gopt": gopt,  # full dict for per-param coverage
            "gdens_norm": gdens_norm,
            "tiles": tiles,
            "ids": ids_for_prev,
            "cam": cam,
            "cam_idx": cam_idx,
        }

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [iter {iteration}] N={current_N} loss={loss.item():.6f} "
                  f"topo={topology_event} ids_next={model._next_gaussian_id}")

    # === Save ===
    print("\n=== Saving outputs ===")

    provenance = {
        "experiment": "r0.2_continuation",
        "semantic_label": "REFERENCE_V1_ABSGRAD",
        "checkpoint_path": checkpoint_path,
        "start_iter": start_iter,
        "n_iters": n_iters,
        "gpu_id": gpu_id,
        "scene": config.scene,
        "N_at_checkpoint": ckpt["num_points"],
        "N_at_end": model._xyz.shape[0],
        "camera_sequence": camera_sequence_path,
        "identity_tracking": "explicit_gaussian_id (Part H)",
        "historical_c51_signal": "xyz grad norm (NOT G_opt_total)",
    }
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    with open(os.path.join(output_dir, "c49_gopt_concentration.json"), "w") as f:
        json.dump({str(k): v for k, v in c49_gopt.items()}, f, indent=2)
    with open(os.path.join(output_dir, "c49_gdens_concentration.json"), "w") as f:
        json.dump({str(k): v for k, v in c49_gdens.items()}, f, indent=2)
    with open(os.path.join(output_dir, "c50_gopt_lag1.json"), "w") as f:
        json.dump({str(k): v for k, v in c50_gopt.items()}, f, indent=2)
    with open(os.path.join(output_dir, "topk_corrected.json"), "w") as f:
        json.dump({str(k): v for k, v in topk_corrected.items()}, f, indent=2)
    with open(os.path.join(output_dir, "coverage_gopt_total.json"), "w") as f:
        json.dump({str(k): v for k, v in coverage_gopt_total.items()}, f, indent=2)
    with open(os.path.join(output_dir, "coverage_gopt_xyz.json"), "w") as f:
        json.dump({str(k): v for k, v in coverage_gopt_xyz.items()}, f, indent=2)
    with open(os.path.join(output_dir, "coverage_per_param.json"), "w") as f:
        json.dump({str(k): v for k, v in coverage_per_param.items()}, f, indent=2)
    with open(os.path.join(output_dir, "gdens_gopt_correlation.json"), "w") as f:
        json.dump({str(k): v for k, v in gdens_gopt_corr.items()}, f, indent=2)
    with open(os.path.join(output_dir, "camera_transitions.json"), "w") as f:
        json.dump({str(k): v for k, v in camera_transitions.items()}, f, indent=2)
    with open(os.path.join(output_dir, "topology_events.json"), "w") as f:
        json.dump({"events": topology_events}, f, indent=2)

    print(f"  Outputs saved to: {output_dir}")
    print(f"  Topology events: {len(topology_events)}")
    print(f"  Final N: {model._xyz.shape[0]}, Final next_id: {model._next_gaussian_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R0.2 Continuation Runner")
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
