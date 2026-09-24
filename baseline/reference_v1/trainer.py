#!/usr/bin/env python3
"""
Reference V1 Trainer — Canonical Room 30K training with full instrumentation.

Produces:
  - provenance.json (complete experiment provenance)
  - camera_sequence.npy (frozen camera index sequence)
  - training_metrics.json (PSNR/SSIM/L1 at eval checkpoints)
  - timing.json (per-phase timing)
  - topology_events.json (clone/split/prune counts per event)
  - lineage_summary.json (Gaussian identity tracker summary)
  - c49_gradient_concentration.json
  - c50_temporal_predictability.json
  - c53_workload_statistics.json

Semantic baseline: REFERENCE_BASELINE_V1
  - Official Graphdeco split (parent removed)
  - Persistent optimizer with state migration
  - View-space mean2D gradient for densification
  - Screen-size + world-size pruning
  - Global opacity reset
  - Fixed SH tensor with active_sh_degree progression
"""

import sys
import os
import json
import math
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Add paths — reference_v1 FIRST so our GaussianModel takes priority
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import our reference GaussianModel BEFORE adding the old scripts path
from gaussian_model import GaussianModel
from config import ReferenceV1Config
from provenance import build_provenance
from instrumentation import (
    GaussianIdentityTracker,
    C49GradientConcentration,
    C50TemporalPredictability,
    C53WorkloadStatistics,
)
from gsplat import rasterization
from colmap_reader import read_points3D_binary, sfm_to_pcd_data

# NOW add the old scripts path for dataset.py (but our GaussianModel is already imported)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "epic05" / "phase7"))
from dataset import GTDataset


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


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def compute_ssim_value(pred, gt, ssim_fn):
    """Return SSIM value (not D-SSIM loss)."""
    return float(1.0 - ssim_fn(pred, gt).item())


def render_with_meta(model, cam, sh_degree, accutile=False):
    """Render and return (image, meta) with means2d retaining grad."""
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
        absgrad=True,  # Required: makes means2d require grad for view-space gradient
        accutile=accutile,
    )
    # Slice means2d to [N, 2] and retain grad for view-space gradient
    # IMPORTANT: call retain_grad on the full tensor, not the slice
    means2d_full = meta["means2d"]  # [1, N, 2]
    if means2d_full.requires_grad:
        means2d_full.retain_grad()
    return r[0].clamp(0, 1), meta, means2d_full


def evaluate_model(model, dataset, ssim_fn, sh_degree, n_cameras=10, accutile=False):
    """Evaluate PSNR/SSIM/L1 on a subset of cameras."""
    psnrs, ssims, l1s = [], [], []
    n_eval = min(n_cameras, len(dataset))
    for i in range(n_eval):
        cam, gt = dataset.get_item(i)
        with torch.no_grad():
            img, _, _ = render_with_meta(model, cam, sh_degree, accutile=accutile)
        psnrs.append(compute_psnr(img, gt))
        ssims.append(compute_ssim_value(img, gt, ssim_fn))
        l1s.append(float(F.l1_loss(img, gt).item()))
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "l1": float(np.mean(l1s)),
        "n_cameras": n_eval,
    }


def compute_scene_extent(dataset, n_samples=50):
    """Compute camera extent (max distance between camera centers)."""
    centers = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        cam = dataset.get_camera(i)
        centers.append(cam.camera_center.cpu().numpy())
    centers = np.array(centers)
    # Extent = max pairwise distance between camera centers
    extent = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            extent = max(extent, d)
    return max(extent, 0.1)  # avoid zero


def run_training(config: ReferenceV1Config, output_dir: str, allow_dirty: bool = False,
                 accutile: bool = False):
    """Run canonical Room 30K training with full instrumentation."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # === Seed ===
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # === Dataset ===
    print(f"Loading dataset: {config.scene}")
    dataset = GTDataset(
        scene=config.scene,
        repo_root=config.repo_root,
        resolution=config.resolution,
        device="cuda",
        background="black",
    )
    n_cameras = len(dataset)
    print(f"  {n_cameras} cameras loaded")

    # === Scene extent ===
    scene_extent = compute_scene_extent(dataset)
    print(f"  Scene extent: {scene_extent:.4f}")

    # === Model initialization ===
    print("Loading SfM point cloud...")
    sfm_path = os.path.join(config.repo_root, "data", "datasets", "mipnerf360",
                            config.scene, "sparse", "0", "points3D.bin")
    if os.path.exists(sfm_path):
        print(f"  Reading COLMAP SfM points from {sfm_path}")
        points3d = read_points3D_binary(sfm_path)
        pcd_data = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)
        print(f"  SfM points: {points3d['num_points']}")
    else:
        # Fallback: use trained checkpoint (document as deviation)
        print(f"  WARNING: SfM file not found at {sfm_path}")
        print(f"  Falling back to trained checkpoint initialization")
        from dataset import load_initial_checkpoint
        pcd_data = load_initial_checkpoint(config.scene, config.repo_root, device="cuda")

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

    # === Provenance ===
    print("Building provenance...")
    # Use SfM path or trained checkpoint path
    if os.path.exists(sfm_path):
        scene_source_path = sfm_path
    else:
        scene_source_path = os.path.join(config.repo_root, "data", "official", "mipnerf360",
                                         config.scene, "point_cloud.ply")
    provenance = build_provenance(
        config=config,
        training_script_path=os.path.abspath(__file__),
        gaussian_model_path=os.path.join(os.path.dirname(__file__), "gaussian_model.py"),
        config_path=os.path.join(os.path.dirname(__file__), "config.py"),
        scene_source_path=scene_source_path,
        initial_gaussian_count=initial_N,
        camera_seed=config.seed,
        allow_dirty=allow_dirty,
    )
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Provenance saved ({len(provenance)} fields)")

    # === Camera sequence ===
    print("Generating camera sequence...")
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
    np.save(os.path.join(output_dir, "camera_sequence.npy"), camera_sequence)
    print(f"  Camera sequence saved ({len(camera_sequence)} iterations)")

    # === SSIM ===
    ssim_fn = SepSSIM(device="cuda")

    # === Instrumentation ===
    identity_tracker = GaussianIdentityTracker(initial_N, device="cuda")
    c49 = C49GradientConcentration()
    c50 = C50TemporalPredictability(device="cuda")
    c53 = C53WorkloadStatistics(device="cuda")

    # === Timing ===
    timing_data = defaultdict(list)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    # === Training metrics ===
    training_metrics = {}
    topology_events = []

    # === Training loop ===
    print(f"\nStarting training: {config.iterations} iterations")
    print(f"  Initial N: {initial_N}")
    print(f"  Scene extent: {scene_extent:.4f}")
    print(f"  Densify: {config.densify_from_iter}-{config.densify_until_iter}, every {config.densification_interval}")
    print(f"  Grad threshold: {config.densify_grad_threshold}")
    print(f"  Opacity reset: every {config.opacity_reset_interval}")

    total_clones = 0
    total_splits = 0
    total_prunes = 0

    for iteration in range(1, config.iterations + 1):
        # Update LR
        model.update_learning_rate(iteration)

        # SH progression (official: every 1000 iters)
        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()

        # Pick camera from frozen sequence
        cam_idx = camera_sequence[iteration - 1]
        cam, gt_image = dataset.get_item(cam_idx)

        # === Forward ===
        iter_start.record()
        fwd_start.record()
        image, meta, means2d = render_with_meta(
            model, cam, model.active_sh_degree, accutile=accutile
        )
        fwd_end.record()
        torch.cuda.synchronize()
        fwd_ms = fwd_start.elapsed_time(fwd_end)

        # === Loss ===
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

        # === Backward ===
        bwd_start.record()
        loss.backward()
        bwd_end.record()
        torch.cuda.synchronize()
        bwd_ms = bwd_start.elapsed_time(bwd_end)

        # === Densification stats (view-space gradient) ===
        radii = meta["radii"][0]  # [N, 2]
        visibility_filter = (radii > 0).any(dim=-1)  # [N]

        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values
            )
            # Use full means2d [1, N, 2], slice gradient after backward
            model.add_densification_stats(means2d, visibility_filter,
                                           width=cam.image_width, height=cam.image_height)

        # === Pre-densification snapshot (for instrumentation) ===
        # Capture gradient stats BEFORE densification resets accumulators
        pre_densify_snapshot = None
        if iteration in config.instrument_iterations:
            with torch.no_grad():
                grad_norms_snap = (model.xyz_gradient_accum / model.denom.clamp_min(1)).squeeze().detach()
                grad_norms_snap[grad_norms_snap.isnan()] = 0
                ids_snap = identity_tracker.get_ids().detach().clone()
                tiles_snap = meta["tiles_per_gauss"][0].float().detach().clone()
                radii_snap = radii.detach().clone()
                scale_norm_snap = model.get_scaling.detach().norm(dim=-1).detach().clone()
                pre_densify_snapshot = {
                    "grad_norms": grad_norms_snap,
                    "ids": ids_snap,
                    "tiles": tiles_snap,
                    "radii": radii_snap,
                    "scale_norm": scale_norm_snap,
                    "N": model._xyz.shape[0],
                }

        # === Topology events ===
        topology_event = None
        if (iteration > config.densify_from_iter and
            iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):

            # Densify and prune
            N_before_densify = model._xyz.shape[0]
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            # Use radii for the current view
            current_radii = radii.float().max(dim=-1).values  # [N]
            if size_threshold is not None:
                event_result = model.densify_and_prune(
                    max_grad=config.densify_grad_threshold,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    max_screen_size=size_threshold,
                    radii=current_radii,
                )
            else:
                # No screen-size pruning before opacity reset
                event_result = model.densify_and_prune(
                    max_grad=config.densify_grad_threshold,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    max_screen_size=None,
                    radii=current_radii,
                )

            # Update identity tracker to match new N
            n_new = event_result["cloned"] + event_result["split"]
            n_pruned = event_result["pruned_total"]
            N_after = model._xyz.shape[0]
            # Rebuild identity tracker to match model size
            # (Exact per-Gaussian tracking through clone/split/prune is complex;
            #  for instrumentation, we track population-level statistics)
            if N_after > N_before_densify:
                identity_tracker.extend(N_after - N_before_densify, iteration, birth_type="clone_or_split")
            elif N_after < N_before_densify:
                # Pruning removed more than densification added
                n_to_remove = N_before_densify - N_after
                keep = torch.ones(N_after + n_to_remove, dtype=torch.bool, device="cuda")
                keep[N_after:] = False
                identity_tracker.filter(keep, iteration)

            topology_event = {
                "iteration": iteration,
                "N_before": N_before_densify,
                "N_after": N_after,
                **event_result,
            }
            topology_events.append(topology_event)
            total_clones += event_result["cloned"]
            total_splits += event_result["split"]
            total_prunes += event_result["pruned_total"]

        # === Opacity reset (official: every opacity_reset_interval) ===
        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()

        # === Optimizer step ===
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        iter_end.record()
        torch.cuda.synchronize()
        total_ms = iter_start.elapsed_time(iter_end)

        # === Timing ===
        timing_data["total_ms"].append(total_ms)
        timing_data["fwd_ms"].append(fwd_ms)
        timing_data["bwd_ms"].append(bwd_ms)

        # === Instrumentation at checkpoints (using pre-densification snapshot) ===
        if iteration in config.instrument_iterations:
            print(f"\n  [ITER {iteration}] Instrumenting...")

            if pre_densify_snapshot is not None:
                grad_norms = pre_densify_snapshot["grad_norms"]
                ids = pre_densify_snapshot["ids"]
                tiles = pre_densify_snapshot["tiles"]
                radii_snap = pre_densify_snapshot["radii"]
                scale_norm = pre_densify_snapshot["scale_norm"]
                snap_N = pre_densify_snapshot["N"]
            else:
                # Fallback: use current state (e.g., iter 500 before densification starts)
                with torch.no_grad():
                    grad_norms = (model.xyz_gradient_accum / model.denom.clamp_min(1)).squeeze().detach()
                    grad_norms[grad_norms.isnan()] = 0
                    ids = identity_tracker.get_ids().detach().clone()
                    tiles = meta["tiles_per_gauss"][0].float().detach().clone()
                    radii_snap = radii.detach().clone()
                    scale_norm = model.get_scaling.detach().norm(dim=-1).detach().clone()
                    snap_N = model._xyz.shape[0]

            # C49: gradient concentration
            print(f"    C49 grad stats: mean={grad_norms.mean().item():.6f} max={grad_norms.max().item():.6f} "
                  f"threshold={config.densify_grad_threshold}")
            c49.record(iteration, grad_norms)
            print(f"    C49 recorded (N={len(grad_norms)})")

            # C50: temporal predictability (use snapshot ids matching grad_norms)
            c50.update(iteration, grad_norms, ids)
            print(f"    C50 updated")

            # C53: workload statistics
            c53.record_checkpoint(iteration, tiles, radii_snap, ids, scale_norm)
            print(f"    C53 recorded")

            # Identity tracker summary
            id_summary = identity_tracker.summary()
            print(f"    Identity: {id_summary}")

        # === Evaluation ===
        if iteration in config.eval_iterations:
            print(f"\n  [ITER {iteration}] Evaluating...")
            metrics = evaluate_model(
                model, dataset, ssim_fn, model.active_sh_degree, accutile=accutile
            )
            metrics["N_gaussians"] = model._xyz.shape[0]
            metrics["iteration"] = iteration
            training_metrics[iteration] = metrics
            print(f"    PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} L1={metrics['l1']:.6f} N={metrics['N_gaussians']}")

        # === Save checkpoint ===
        if iteration in config.checkpoint_iterations:
            ckpt = model.capture()
            torch.save(ckpt, os.path.join(output_dir, "checkpoints", f"iter_{iteration}.pt"))
            print(f"  [ITER {iteration}] Checkpoint saved")

        # === Progress ===
        if iteration % 500 == 0:
            ema_loss = 0.4 * loss.item() + 0.6 * timing_data.get("ema_loss", [0])[-1] if timing_data.get("ema_loss") else loss.item()
            timing_data["ema_loss"].append(ema_loss)
            print(f"  [ITER {iteration}] Loss={loss.item():.6f} N={model._xyz.shape[0]} "
                  f"fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms total={total_ms:.1f}ms")

    # === Save outputs ===
    print("\n=== Saving outputs ===")

    # Training metrics
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump({"checkpoints": {str(k): v for k, v in training_metrics.items()},
                   "final_N": model._xyz.shape[0],
                   "total_clones": total_clones,
                   "total_splits": total_splits,
                   "total_prunes": total_prunes}, f, indent=2)

    # Timing
    timing_summary = {}
    total_iters = len(timing_data["total_ms"])
    for phase_name, (start, end) in [("0-5K", (0, 5000)), ("5-10K", (5000, 10000)),
                                      ("10-15K", (10000, 15000)), ("15-20K", (15000, 20000)),
                                      ("20-25K", (20000, 25000)), ("25-30K", (25000, 30000))]:
        idx_start = min(start, total_iters)
        idx_end = min(end, total_iters)
        if idx_end > idx_start:
            timing_summary[phase_name] = {
                "total_ms_mean": float(np.mean(timing_data["total_ms"][idx_start:idx_end])),
                "total_ms_median": float(np.median(timing_data["total_ms"][idx_start:idx_end])),
                "fwd_ms_mean": float(np.mean(timing_data["fwd_ms"][idx_start:idx_end])),
                "bwd_ms_mean": float(np.mean(timing_data["bwd_ms"][idx_start:idx_end])),
            }
        else:
            timing_summary[phase_name] = {"total_ms_mean": 0, "total_ms_median": 0,
                                          "fwd_ms_mean": 0, "bwd_ms_mean": 0}
    timing_summary["full_run"] = {
        "total_ms_mean": float(np.mean(timing_data["total_ms"])),
        "total_ms_median": float(np.median(timing_data["total_ms"])),
        "fwd_ms_mean": float(np.mean(timing_data["fwd_ms"])),
        "bwd_ms_mean": float(np.mean(timing_data["bwd_ms"])),
    }
    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump(timing_summary, f, indent=2)

    # Topology events
    with open(os.path.join(output_dir, "topology_events.json"), "w") as f:
        json.dump({"events": topology_events,
                   "total_clones": total_clones,
                   "total_splits": total_splits,
                   "total_prunes": total_prunes}, f, indent=2)

    # Lineage summary
    with open(os.path.join(output_dir, "lineage_summary.json"), "w") as f:
        json.dump(identity_tracker.summary(), f, indent=2)

    # C49/C50/C53
    with open(os.path.join(output_dir, "c49_gradient_concentration.json"), "w") as f:
        json.dump(c49.to_dict(), f, indent=2)
    with open(os.path.join(output_dir, "c50_temporal_predictability.json"), "w") as f:
        json.dump(c50.to_dict(), f, indent=2)
    with open(os.path.join(output_dir, "c53_workload_statistics.json"), "w") as f:
        json.dump(c53.to_dict(), f, indent=2)

    # Config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print("\n=== Training complete ===")
    print(f"  Final N: {model._xyz.shape[0]}")
    print(f"  Total clones: {total_clones}")
    print(f"  Total splits: {total_splits}")
    print(f"  Total prunes: {total_prunes}")
    if config.iterations in training_metrics:
        m = training_metrics[config.iterations]
        print(f"  Final PSNR: {m['psnr']:.2f}")
        print(f"  Final SSIM: {m['ssim']:.4f}")
    print(f"  Mean iter time: {timing_summary['full_run']['total_ms_mean']:.1f}ms")
    print(f"\n  Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference V1 Trainer")
    parser.add_argument("--scene", default="room", type=str)
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--allow_dirty", action="store_true",
                       help="Allow dirty git state (dev only)")
    parser.add_argument("--accutile", choices=("off", "on"), default="off",
                        help="Use conservative ellipse-tile intersection (default: off).")
    args = parser.parse_args()

    config = ReferenceV1Config(
        scene=args.scene,
        iterations=args.iterations,
    )
    if args.output_dir is None:
        args.output_dir = f"results/reference_v1/{args.scene}_{args.iterations // 1000}k"

    run_training(
        config, args.output_dir, allow_dirty=args.allow_dirty,
        accutile=(args.accutile == "on"),
    )
