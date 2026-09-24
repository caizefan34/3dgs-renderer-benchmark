"""
R0.1 Continuation Runner — Exact evidence collection for C49/C50/C53 recalibration.

Runs 200 consecutive iterations from a checkpoint, collecting at EVERY iteration:
  - G_opt: per-Gaussian optimization gradient norms (xyz, opacity, scale, rotation, SH, total)
  - G_dens: per-Gaussian densification gradient (means2d.absgrad, pixel-space scaled)
  - Workload: tiles_per_gauss
  - Camera transition: center distance, view-direction angle
  - Exact identity tracking through clone/split/prune via xyz hash matching

Does NOT retrain. Does NOT implement new optimization. Uses frozen REFERENCE_V1_ABSGRAD code.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 continuation_runner.py \
    --checkpoint results/reference_v1/ckpt_14k/checkpoints/iter_2000.pt \
    --start-iter 2000 --n-iters 200 \
    --output results/reference_v1/r0.1/window_2000
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

# === Path setup (import reference_v1 FIRST to avoid shadowing) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline", "reference_v1")
SRC_DIR = os.path.join(REPO_ROOT, "src")

sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, SRC_DIR)

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization

# NOW add old scripts path for dataset.py (after our GaussianModel is imported)
OLD_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "epic05", "phase7")
sys.path.insert(0, OLD_SCRIPTS)
from dataset import GTDataset


# === SSIM (exact copy from trainer.py for gradient consistency) ===
class SepSSIM:
    """Separable SSIM — mathematically equivalent to standard SSIM."""

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


# === Exact Identity Tracker ===
class ExactIdentityTracker:
    """Track stable Gaussian IDs through clone/split/prune using xyz hash matching.

    Rules:
      - Survivor keeps ID
      - Clone gets new child ID
      - Split parent dies
      - Split children get new IDs
      - Pruned ID dies

    Uses exact xyz matching (no optimizer step between densification and match).
    """

    def __init__(self, n_initial: int):
        self.next_id = n_initial
        self.index_to_id = list(range(n_initial))
        self.birth_iter = {i: 0 for i in range(n_initial)}
        self.birth_type = {i: "initial" for i in range(n_initial)}
        self.death_iter = {}

    def get_ids(self):
        return list(self.index_to_id)

    def get_ids_tensor(self, device="cuda"):
        return torch.tensor(self.index_to_id, dtype=torch.long, device=device)

    def reconcile(self, xyz_before, ids_before, xyz_after, iteration):
        """Reconcile IDs after a topology change using xyz hash matching.

        Args:
            xyz_before: [N_before, 3] tensor (pre-densification positions)
            ids_before: list of int (pre-densification stable IDs)
            xyz_after: [N_after, 3] tensor (post-densification positions)
            iteration: current iteration number

        Returns:
            ids_after: list of int (post-densification stable IDs)
        """
        N_before = len(ids_before)
        N_after = xyz_after.shape[0]

        # Build hash map: rounded xyz -> list of before-indices
        xyz_hash = {}
        for i in range(N_before):
            key = tuple(xyz_before[i].cpu().tolist())
            key = tuple(round(v, 6) for v in key)
            if key not in xyz_hash:
                xyz_hash[key] = []
            xyz_hash[key].append(i)

        # Match each after-Gaussian
        ids_after = []
        used_before = set()

        for i in range(N_after):
            key = tuple(xyz_after[i].cpu().tolist())
            key = tuple(round(v, 6) for v in key)

            matched = False
            if key in xyz_hash:
                for before_idx in xyz_hash[key]:
                    if before_idx not in used_before:
                        # Survivor
                        ids_after.append(ids_before[before_idx])
                        used_before.add(before_idx)
                        matched = True
                        break

            if not matched:
                # New Gaussian (clone child or split child)
                new_id = self.next_id
                ids_after.append(new_id)
                self.birth_iter[new_id] = iteration
                self.birth_type[new_id] = "clone_or_split_child"
                self.next_id += 1

        # Record deaths for unmatched before-Gaussians
        for i in range(N_before):
            if i not in used_before:
                self.death_iter[ids_before[i]] = iteration

        self.index_to_id = ids_after
        return ids_after


# === Camera transition metrics ===
def compute_camera_transition(cam_t, cam_t1):
    """Compute transition metrics between two cameras.

    Returns:
        center_dist: ||center_t - center_{t+1}||
        view_angle: angle between view directions (degrees)
    """
    center_t = cam_t.camera_center
    center_t1 = cam_t1.camera_center
    center_dist = float(torch.norm(center_t - center_t1).item())

    # View direction: -Z column of view matrix (camera forward)
    view_dir_t = -cam_t.viewmatrix[:3, 2]
    view_dir_t1 = -cam_t1.viewmatrix[:3, 2]
    cos_angle = float(torch.dot(view_dir_t, view_dir_t1).item())
    cos_angle = max(-1.0, min(1.0, cos_angle))
    view_angle = math.degrees(math.acos(cos_angle))

    return center_dist, view_angle


# === Render function (matching trainer) ===
def render_with_meta(model, cam, sh_degree):
    """Render and return (image, meta, means2d) with means2d retaining grad."""
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
    means2d_full = meta["means2d"]  # [1, N, 2]
    means2d_full.retain_grad()
    return r, meta, means2d_full


# === G_opt extraction ===
def extract_gopt(model):
    """Extract per-Gaussian optimization gradient norms after backward().

    Returns dict of [N] float32 CPU tensors:
        xyz_norm: ||dL/dxyz_i||_2
        opacity_abs: |dL/dopacity_i|
        scale_norm: ||dL/dscaling_i||_2
        rot_norm: ||dL/drotation_i||_2
        shs_norm: ||dL/dshs_i||_2 (flattened over K*3)
        total_norm: sqrt(sum of squared norms above)
    """
    grads = {}

    # xyz [N, 3]
    if model._xyz.grad is not None:
        grads["xyz_norm"] = model._xyz.grad.norm(dim=-1).detach()
    else:
        grads["xyz_norm"] = torch.zeros(model._xyz.shape[0], device="cuda")

    # opacity [N]
    if model._opacity.grad is not None:
        grads["opacity_abs"] = model._opacity.grad.abs().detach()
    else:
        grads["opacity_abs"] = torch.zeros(model._opacity.shape[0], device="cuda")

    # scaling [N, 3]
    if model._scaling.grad is not None:
        grads["scale_norm"] = model._scaling.grad.norm(dim=-1).detach()
    else:
        grads["scale_norm"] = torch.zeros(model._scaling.shape[0], device="cuda")

    # rotation [N, 4]
    if model._rotation.grad is not None:
        grads["rot_norm"] = model._rotation.grad.norm(dim=-1).detach()
    else:
        grads["rot_norm"] = torch.zeros(model._rotation.shape[0], device="cuda")

    # shs [N, K, 3]
    if model._shs.grad is not None:
        grads["shs_norm"] = model._shs.grad.flatten(start_dim=1).norm(dim=-1).detach()
    else:
        grads["shs_norm"] = torch.zeros(model._shs.shape[0], device="cuda")

    # Total (combined importance for C51)
    total_sq = (grads["xyz_norm"]**2 + grads["opacity_abs"]**2 +
                grads["scale_norm"]**2 + grads["rot_norm"]**2 + grads["shs_norm"]**2)
    grads["total_norm"] = torch.sqrt(total_sq + 1e-20)

    # Move to CPU
    return {k: v.cpu().numpy().astype(np.float32) for k, v in grads.items()}


# === G_dens extraction ===
def extract_gdens(means2d, visibility_filter, width, height):
    """Extract per-Gaussian densification gradient (means2d.absgrad, pixel-space).

    Returns:
        gdens_norm: [N] float32 CPU — L2 norm of pixel-space absgrad per Gaussian
        gdens_visibility: [N] bool CPU — whether Gaussian was visible
    """
    grad = getattr(means2d, "absgrad", None)
    if grad is None:
        grad = means2d.grad
    if grad is None:
        N = means2d.shape[1] if means2d.dim() == 3 else means2d.shape[0]
        return np.zeros(N, dtype=np.float32), np.zeros(N, dtype=bool)

    if grad.dim() == 3:
        grad = grad[0]  # [N, 2]

    # Scale to pixel space
    grad = grad.clone()
    grad[:, 0] *= width / 2.0
    grad[:, 1] *= height / 2.0

    gdens_norm = grad.norm(dim=-1).detach().cpu().numpy().astype(np.float32)
    gdens_vis = visibility_filter.detach().cpu().numpy().astype(bool)
    return gdens_norm, gdens_vis


# === Concentration metrics (C49) ===
def compute_concentration(values, visible_mask=None):
    """Compute concentration metrics for a gradient/workload signal.

    Args:
        values: [N] array of per-Gaussian values
        visible_mask: [N] bool array, if provided only visible Gaussians are considered

    Returns:
        dict with gini, top1/5/10/20/32/50_mass, mean, median, max, n
    """
    if visible_mask is not None:
        vals = values[visible_mask]
    else:
        vals = values

    n = len(vals)
    if n == 0:
        return {"gini": 0, "top1_mass": 0, "top5_mass": 0, "top10_mass": 0,
                "top20_mass": 0, "top32_mass": 0, "top50_mass": 0,
                "mean": 0, "median": 0, "max": 0, "n": 0}

    total = vals.sum()
    if total < 1e-20:
        return {"gini": 0, "top1_mass": 0, "top5_mass": 0, "top10_mass": 0,
                "top20_mass": 0, "top32_mass": 0, "top50_mass": 0,
                "mean": 0, "median": 0, "max": 0, "n": n}

    sorted_vals = np.sort(vals)[::-1]  # descending
    cumsum = np.cumsum(sorted_vals)
    masses = {}
    for pct in [1, 5, 10, 20, 32, 50]:
        k = max(1, int(n * pct / 100))
        masses[f"top{pct}_mass"] = float(cumsum[k-1] / total)

    # Gini coefficient
    sorted_all = np.sort(vals)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_all) / (n * total)) - (n + 1) / n)

    return {
        **masses,
        "gini": gini,
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "max": float(vals.max()),
        "n": n,
    }


# === Lag-1 metrics (C50/C53) ===
def compute_lag1_metrics(vals_t, vals_t1, ids_t, ids_t1):
    """Compute lag-1 temporal metrics for matched Gaussians.

    Args:
        vals_t: [N_t] array of values at iteration t
        vals_t1: [N_t1] array of values at iteration t+1
        ids_t: [N_t] array of stable IDs at iteration t
        ids_t1: [N_t1] array of stable IDs at iteration t+1

    Returns:
        dict with pearson, spearman, top-K jaccard, recall@K, n_matched
    """
    # Match by stable IDs
    id_to_idx_t1 = {int(gid): i for i, gid in enumerate(ids_t1)}

    matched_t = []
    matched_t1 = []
    for i, gid in enumerate(ids_t):
        gid = int(gid)
        if gid in id_to_idx_t1:
            matched_t.append(vals_t[i])
            matched_t1.append(vals_t1[id_to_idx_t1[gid]])

    n_matched = len(matched_t)
    if n_matched < 10:
        return {"pearson": 0, "spearman": 0, "n_matched": n_matched,
                "top1_jaccard": 0, "top5_jaccard": 0, "top10_jaccard": 0,
                "top20_jaccard": 0, "top32_jaccard": 0, "top50_jaccard": 0,
                "recall1": 0, "recall5": 0, "recall10": 0, "recall20": 0}

    arr_t = np.array(matched_t)
    arr_t1 = np.array(matched_t1)

    # Pearson
    if np.std(arr_t) > 1e-10 and np.std(arr_t1) > 1e-10:
        pearson = float(np.corrcoef(arr_t, arr_t1)[0, 1])
    else:
        pearson = 0.0

    # Spearman (rank correlation)
    rank_t = np.argsort(np.argsort(arr_t)).astype(float)
    rank_t1 = np.argsort(np.argsort(arr_t1)).astype(float)
    if np.std(rank_t) > 1e-10 and np.std(rank_t1) > 1e-10:
        spearman = float(np.corrcoef(rank_t, rank_t1)[0, 1])
    else:
        spearman = 0.0

    n = n_matched
    result = {"pearson": pearson, "spearman": spearman, "n_matched": n_matched}

    # Top-K Jaccard and Recall
    for k_pct in [1, 5, 10, 20, 32, 50]:
        k = max(1, int(n * k_pct / 100))
        top_t = set(np.argsort(arr_t)[::-1][:k])
        top_t1 = set(np.argsort(arr_t1)[::-1][:k])
        jaccard = len(top_t & top_t1) / k
        result[f"top{k_pct}_jaccard"] = float(jaccard)

    # Recall@K (fraction of top-K at t that are in top-K at t+1)
    for k_pct in [10, 20]:
        k = max(1, int(n * k_pct / 100))
        top_t = set(np.argsort(arr_t)[::-1][:k])
        top_t1 = set(np.argsort(arr_t1)[::-1][:k])
        recall = len(top_t & top_t1) / k
        result[f"recall{k_pct}"] = float(recall)

    return result


# === Main continuation runner ===
def run_continuation(checkpoint_path, start_iter, n_iters, output_dir,
                     config, camera_sequence_path, gpu_id=0):
    """Run 200-iter continuation from checkpoint with full instrumentation."""

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"

    # === Load dataset ===
    print(f"Loading dataset: {config.scene}")
    dataset = GTDataset(config.scene, config.repo_root)
    n_cameras = len(dataset)
    # Compute scene extent (max pairwise camera center distance)
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

    # === Load camera sequence (from 30K run) ===
    camera_sequence = np.load(camera_sequence_path)
    print(f"  Camera sequence: {len(camera_sequence)} entries")

    # === Load checkpoint ===
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint N: {ckpt['num_points']}, SH degree: {ckpt['active_sh_degree']}")

    # === Create model and restore from checkpoint ===
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

    # === Identity tracker ===
    id_tracker = ExactIdentityTracker(N)

    # === SSIM ===
    ssim_fn = SepSSIM(device=device)

    # === Data storage ===
    # Per-iteration concentration stats (small)
    c49_gopt = {}  # iter -> concentration dict
    c49_gdens = {}  # iter -> concentration dict

    # Per-pair lag-1 metrics (small)
    c50_gopt = {}  # iter -> lag1 dict
    c50_gdens = {}  # iter -> lag1 dict
    c53_lag1 = {}  # iter -> lag1 dict

    # Camera transitions
    camera_transitions = {}  # iter -> {center_dist, view_angle}

    # Topology events
    topology_events = []

    # Previous iteration data (for lag-1 computation)
    prev_data = None  # dict with gopt, gdens, tiles, ids, cam_idx

    # === Run continuation ===
    print(f"\nStarting continuation: {n_iters} iterations (from iter {start_iter+1})")
    print(f"  Densification at iters where (iter % 100 == 0) and (500 < iter < 15000)")

    for i in range(n_iters):
        iteration = start_iter + 1 + i

        # Pick camera from frozen sequence
        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)

        # Update LR
        model.update_learning_rate(iteration)

        # === Forward ===
        image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree)
        # gsplat returns [C, H, W, 3] — squeeze to [H, W, 3] for SSIM
        if image.ndim == 4:
            image = image.squeeze(0)  # [H, W, 3]
        radii = meta["radii"][0]  # [N, 2]
        visibility_filter = (radii > 0).any(dim=-1)  # [N]

        # === Loss ===
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

        # === Backward ===
        loss.backward()

        # === Extract G_opt (per-Gaussian optimization gradients) ===
        gopt = extract_gopt(model)

        # === Extract G_dens (densification gradient) ===
        gdens_norm, gdens_vis = extract_gdens(
            means2d, visibility_filter, cam.image_width, cam.image_height
        )

        # === Extract workload ===
        tiles = meta["tiles_per_gauss"][0].float().detach().cpu().numpy().astype(np.float32)

        # === Current IDs ===
        # Save IDs BEFORE any densification — these match the gradient data
        ids_for_prev = list(id_tracker.get_ids())  # copy, matches gopt/gdens/tiles size
        current_N = model._xyz.shape[0]

        # === Compute C49 concentration (per iteration) ===
        # G_opt: use total_norm for primary signal, also compute per-parameter
        c49_gopt[iteration] = compute_concentration(gopt["total_norm"], gdens_vis)
        # Also compute for individual params
        for pname in ["xyz_norm", "opacity_abs", "scale_norm", "rot_norm", "shs_norm"]:
            c49_gopt[iteration][f"_{pname}"] = compute_concentration(gopt[pname], gdens_vis)

        # G_dens: use only visible Gaussians
        c49_gdens[iteration] = compute_concentration(gdens_norm, gdens_vis)

        # === Compute lag-1 metrics (if we have previous iteration data) ===
        if prev_data is not None:
            # Camera transition
            center_dist, view_angle = compute_camera_transition(
                prev_data["cam"], cam
            )
            camera_transitions[iteration] = {
                "center_dist": center_dist,
                "view_angle": view_angle,
                "cam_t": prev_data["cam_idx"],
                "cam_t1": cam_idx,
            }

            # C50: G_opt lag-1
            c50_gopt[iteration] = compute_lag1_metrics(
                prev_data["gopt_total"], gopt["total_norm"],
                prev_data["ids"], ids_for_prev
            )

            # C50: G_dens lag-1
            c50_gdens[iteration] = compute_lag1_metrics(
                prev_data["gdens_norm"], gdens_norm,
                prev_data["ids"], ids_for_prev
            )

            # C53: Workload lag-1
            c53_lag1[iteration] = compute_lag1_metrics(
                prev_data["tiles"], tiles,
                prev_data["ids"], ids_for_prev
            )

        # === Densification stats (for the model's internal densification) ===
        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values
            )
            model.add_densification_stats(means2d, visibility_filter,
                                           width=cam.image_width, height=cam.image_height)

        # === Topology events ===
        topology_event = "none"
        if (iteration > config.densify_from_iter and
            iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):

            # Save pre-densification state for identity reconciliation
            xyz_before = model._xyz.detach().clone()
            ids_before = id_tracker.get_ids()
            N_before = model._xyz.shape[0]

            # Densify and prune
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

            # Reconcile identities using xyz matching
            xyz_after = model._xyz.detach()
            id_tracker.reconcile(xyz_before, ids_before, xyz_after, iteration)

            topology_event = "clone_split"
            topology_events.append({
                "iteration": iteration,
                "N_before": N_before,
                "N_after": model._xyz.shape[0],
                **event_result,
            })

            # NOTE: current_N updated but ids_for_prev already saved (matches gradient data)
            current_N = model._xyz.shape[0]

        # === Opacity reset ===
        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()
            topology_event = "prune_reset" if topology_event == "none" else topology_event

        # === Optimizer step ===
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        # === Store previous iteration data ===
        # IMPORTANT: ids_for_prev matches the size of gopt/gdens/tiles (pre-densification)
        prev_data = {
            "gopt_total": gopt["total_norm"],
            "gdens_norm": gdens_norm,
            "tiles": tiles,
            "ids": ids_for_prev,
            "cam": cam,
            "cam_idx": cam_idx,
        }

        # === Progress ===
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [iter {iteration}] N={current_N} loss={loss.item():.6f} "
                  f"topo={topology_event}")

    # === Save outputs ===
    print("\n=== Saving outputs ===")

    # Provenance
    provenance = {
        "experiment": "r0.1_continuation",
        "semantic_label": "REFERENCE_V1_ABSGRAD",
        "checkpoint_path": checkpoint_path,
        "start_iter": start_iter,
        "n_iters": n_iters,
        "gpu_id": gpu_id,
        "scene": config.scene,
        "N_at_checkpoint": ckpt["num_points"],
        "N_at_end": model._xyz.shape[0],
        "camera_sequence": camera_sequence_path,
    }
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    # C49 concentration
    with open(os.path.join(output_dir, "c49_gopt_concentration.json"), "w") as f:
        json.dump({str(k): v for k, v in c49_gopt.items()}, f, indent=2)
    with open(os.path.join(output_dir, "c49_gdens_concentration.json"), "w") as f:
        json.dump({str(k): v for k, v in c49_gdens.items()}, f, indent=2)

    # C50 lag-1
    with open(os.path.join(output_dir, "c50_true_lag1_gopt.json"), "w") as f:
        json.dump({str(k): v for k, v in c50_gopt.items()}, f, indent=2)
    with open(os.path.join(output_dir, "c50_true_lag1_gdens.json"), "w") as f:
        json.dump({str(k): v for k, v in c50_gdens.items()}, f, indent=2)

    # C53 lag-1
    with open(os.path.join(output_dir, "c53_true_lag1_workload.json"), "w") as f:
        json.dump({str(k): v for k, v in c53_lag1.items()}, f, indent=2)

    # Camera transitions
    with open(os.path.join(output_dir, "camera_transitions.json"), "w") as f:
        json.dump({str(k): v for k, v in camera_transitions.items()}, f, indent=2)

    # Topology events
    with open(os.path.join(output_dir, "topology_events.json"), "w") as f:
        json.dump({"events": topology_events}, f, indent=2)

    print(f"  Outputs saved to: {output_dir}")
    print(f"  Topology events: {len(topology_events)}")
    print(f"  Final N: {model._xyz.shape[0]}")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R0.1 Continuation Runner")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--start-iter", type=int, required=True, help="Checkpoint iteration")
    parser.add_argument("--n-iters", type=int, default=200, help="Number of continuation iterations")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--camera-sequence", required=True, help="Path to camera_sequence.npy")
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
