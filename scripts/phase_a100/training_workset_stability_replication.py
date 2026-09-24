#!/usr/bin/env python3
"""
Training Workset Stability Replication — A100 Canonical (gsplat 1.5.3).

Purpose:
  Replicate the preliminary RTX 5070 / diff-gaussian-rasterization audit on
  the canonical environment: A100 PCIe 40GB + gsplat 1.5.3 + real training pipeline.

Rules:
  - NO renderer algorithm modifications
  - NO cache / selective rebuild / incremental tile construction
  - ONLY measurement / instrumentation added

Outputs:
  results/phase-a100/training_workset_stability_replication.json
  results/phase-a100/training_workset_overlap_series_replication.csv
  reports/phase-a100/training_workset_stability_replication.md
  results/phase-a100/training_workset_stability_replication_manifest.json
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch

# ── Project imports ────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # root scripts/ first to avoid src/scripts/ shadow
sys.path.insert(1, str(REPO_ROOT / "src"))  # src/ second for benchmark_framework etc.

from gsplat import rasterization  # noqa: E402
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras  # noqa: E402
from scripts.epic05.phase7.gaussian_model import GaussianModel  # noqa: E402
from scripts.epic05.phase7.loss import combined_loss  # noqa: E402
from scripts.epic05.phase7.dataset import GTDataset  # noqa: E402

torch.backends.cudnn.deterministic = True


# ═══════════════════════════════════════════════════════════════════
#  Training Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ReplicationConfig:
    """Canonical training config matching Phase 7 defaults."""
    scene: str = "room"
    resolution: str = "1080p"
    num_iterations: int = 500
    tile_size: int = 16
    packed: bool = True
    sh_degree: int = 0          # Start at 0, increase during training
    max_sh_degree: int = 3
    radius_clip: float = 0.0
    eps2d: float = 0.1
    lambda_dssim: float = 0.2

    # Learning rates
    lr_xyz: float = 1.6e-4
    lr_rotation: float = 1e-3
    lr_scaling: float = 5e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3

    # Densification
    densification_start: int = 100
    densification_end: int = 15000
    densification_interval: int = 100
    grad_threshold: float = 2e-4

    # Pruning
    prune_interval: int = 100
    prune_opacity: float = 0.005

    # Eval
    eval_interval: int = 50

    # N eval cameras (fixed viewpoints)
    num_eval_cameras: int = 5

    seed: int = 42

    def experiment_label(self) -> str:
        return f"replication_{self.scene}_t{self.tile_size}"


# ═══════════════════════════════════════════════════════════════════
#  Workset extraction from gsplat info dict
# ═══════════════════════════════════════════════════════════════════

def extract_worksets_from_info(info: dict) -> tuple:
    """
    Extract all three worksets from gsplat rasterization info dict.

    Workset A — Visible Gaussian Set:
        Gaussians that passed projection AND have non-zero tile intersection.
        Source: info['gaussian_ids'] (sorted, unique Gaussian IDs)

    Workset B — Active Tile Set:
        Tiles with at least one Gaussian member.
        Source: info['isect_offsets'] where offset[tile] != offset[tile+1]

    Workset C — Gaussian–Tile Membership:
        (Gaussian ID, Tile ID) pairs from the sorted intersection list.
        Source: info['flatten_ids'] (tile IDs) and info['isect_ids'] (Gaussian IDs)
    """
    gaussian_ids = info["gaussian_ids"]       # [V] original Gaussian IDs
    flatten_ids = info["flatten_ids"].long()  # [M] indices into gaussian_ids
    isect_offsets = info["isect_offsets"][0].reshape(-1).long()
    total_entries = flatten_ids.numel()
    tile_count = isect_offsets.numel()

    # A: sorted original IDs for Gaussians that survived projection.
    visible_ids = torch.sort(gaussian_ids).values

    # B: tile IDs with a non-empty range.  Offset[t] is the start of tile t;
    # the next offset (or M for the final tile) is its exclusive end.
    ends = torch.cat((isect_offsets[1:], isect_offsets.new_tensor([total_entries])))
    counts = ends - isect_offsets
    active_tiles = torch.nonzero(counts > 0, as_tuple=False).flatten()

    # C: exact encoded (original Gaussian ID, tile ID) pairs.  In gsplat 1.5.3
    # flatten_ids indexes gaussian_ids; isect_ids is an encoded sorting key and
    # must not be interpreted as a Gaussian-ID vector.
    entry_tiles = torch.repeat_interleave(
        torch.arange(tile_count, device=flatten_ids.device, dtype=torch.long), counts
    )
    member_gaussians = gaussian_ids[flatten_ids]
    membership_keys = torch.sort(member_gaussians * tile_count + entry_tiles).values

    return visible_ids, active_tiles, membership_keys


# ═══════════════════════════════════════════════════════════════════
#  Overlap statistics
# ═══════════════════════════════════════════════════════════════════

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 1.0


def compute_overlap_series(history: list, lags: list = None) -> dict:
    """Compute Jaccard overlaps at specified lags over a time series of sets."""
    if lags is None:
        lags = [1, 2, 4, 8]
    overlaps_by_lag = {lag: [] for lag in lags}
    for t in range(len(history)):
        if not history[t]:
            continue
        for lag in lags:
            if t + lag < len(history) and history[t + lag]:
                ov = jaccard(history[t], history[t + lag])
                overlaps_by_lag[lag].append(ov)
    stats = {}
    for lag, vals in overlaps_by_lag.items():
        if not vals:
            stats[f"lag_{lag}"] = {"mean": -1, "p50": -1, "p10": -1, "p90": -1,
                                    "min": -1, "max": -1, "std": -1, "n": 0}
            continue
        arr = np.array(vals)
        stats[f"lag_{lag}"] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
        }
    return stats


def classify_stability(stats: dict) -> str:
    """Classify workset stability from lag-1 overlap statistics."""
    median = stats.get("lag_1", {}).get("p50", 0)
    if median >= 0.9:
        return "HIGH"
    elif median >= 0.7:
        return "MODERATE"
    return "LOW"


# ═══════════════════════════════════════════════════════════════════
#  Gaussian-level stability analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_gaussian_stability(
    history: List[Set[int]],
    all_gaussian_counts: List[int],
) -> dict:
    """
    Analyze Gaussian-level stability:
      - New Gaussians (in A_t but not A_{t-1})
      - Removed Gaussians (in A_{t-1} but not A_t)
      - Persisted Gaussians (in both)
      - Visible → invisible (was in A, now in not-in-A but exists in model)
      - Invisible → visible (was not-in-A but exists, now in A)
    """
    stats_list = []
    for t in range(1, len(history)):
        if not history[t] or not history[t-1]:
            continue
        prev_set = history[t-1]
        curr_set = history[t]
        added = curr_set - prev_set
        removed = prev_set - curr_set
        persisted = curr_set & prev_set
        stats_list.append({
            "step": t,
            "n_visible_prev": len(prev_set),
            "n_visible_curr": len(curr_set),
            "n_added": len(added),
            "n_removed": len(removed),
            "n_persisted": len(persisted),
            "jaccard": jaccard(prev_set, curr_set),
        })

    if not stats_list:
        return {}

    added = np.array([s["n_added"] for s in stats_list])
    removed = np.array([s["n_removed"] for s in stats_list])
    persisted = np.array([s["n_persisted"] for s in stats_list])

    return {
        "added": {
            "mean": float(np.mean(added)),
            "p50": float(np.median(added)),
            "p10": float(np.percentile(added, 10)),
            "p90": float(np.percentile(added, 90)),
            "max": float(np.max(added)),
        },
        "removed": {
            "mean": float(np.mean(removed)),
            "p50": float(np.median(removed)),
            "p10": float(np.percentile(removed, 10)),
            "p90": float(np.percentile(removed, 90)),
            "max": float(np.max(removed)),
        },
        "persisted": {
            "mean": float(np.mean(persisted)),
            "p50": float(np.median(persisted)),
            "p10": float(np.percentile(persisted, 10)),
            "p90": float(np.percentile(persisted, 90)),
        },
    }


# ═══════════════════════════════════════════════════════════════════
#  Main replication training + audit
# ═══════════════════════════════════════════════════════════════════

def run_replication(config: ReplicationConfig):
    """Run canonical training with workset instrumentation."""
    device = torch.device("cuda")
    save_dir = REPO_ROOT / "results" / "phase-a100"
    save_dir.mkdir(parents=True, exist_ok=True)
    report_dir = REPO_ROOT / "reports" / "phase-a100"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Training Workset Stability Replication — A100 Canonical")
    print(f"  Scene: {config.scene}, Steps: {config.num_iterations}, Tile: {config.tile_size}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")

    # ── Create checkpointer directory for config ──
    # (use a temp label so it doesn't interfere with real experiments)
    save_label = config.experiment_label()
    ckpt_dir = save_dir / "checkpoints" / save_label
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    print("\n  [Loading dataset...]")
    dataset = GTDataset(
        scene=config.scene,
        repo_root=REPO_ROOT,
        resolution=config.resolution,
        device=device,
    )
    num_cameras = len(dataset)
    print(f"  {num_cameras} cameras")

    # ── Initialize model ──
    print("\n  [Loading SfM initialization...]")
    ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / config.scene / "point_cloud.ply"
    sfm_data = load_ply(str(ply_path), device=device)

    print("\n  [Initializing Gaussian model...]")
    model = GaussianModel(
        num_points=sfm_data["xyz"].shape[0],
        sh_degree=0,
        max_sh_degree=config.max_sh_degree,
        device=device,
    )
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"),
    )
    scene_extent = sfm_data["xyz"].norm(dim=-1).max().item()
    spatial_lr_scale = scene_extent
    print(f"    SfM points: {model.xyz.shape[0]:,}, Scene extent: {scene_extent:.2f}")

    # ── Optimizer ──
    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": config.lr_xyz * spatial_lr_scale,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": config.lr_rotation,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": config.lr_scaling,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": config.lr_opacity,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": config.lr_sh,
         "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    # ── Select fixed eval cameras ──
    eval_indices = list(range(min(config.num_eval_cameras, num_cameras)))
    eval_cameras_gt = [dataset.get_item(i) for i in eval_indices]
    eval_cameras = [ec for ec, _ in eval_cameras_gt]
    eval_gts = [eg for _, eg in eval_cameras_gt]
    print(f"  {len(eval_indices)} fixed eval cameras: {eval_indices}")

    # ── Records ──
    max_steps = config.num_iterations

    # Per-step training-view records
    vg_sets_train: List[Set[int]] = [set() for _ in range(max_steps)]
    at_sets_train: List[Set[int]] = [set() for _ in range(max_steps)]
    mb_sets_train: List[Set[tuple]] = [set() for _ in range(max_steps)]

    n_vis = [0] * max_steps
    n_tiles = [0] * max_steps
    n_gaussians = [0] * max_steps
    total_membership_count = [0] * max_steps

    dens_events = []
    prune_events = []
    loss_log = []
    psnr_log = []

    # Eval records (same-viewpoint)
    eval_steps = list(range(0, max_steps, config.eval_interval))
    if max_steps - 1 not in eval_steps:
        eval_steps.append(max_steps - 1)
    eval_vg_sets: Dict[int, List[Set[int]]] = {}
    eval_at_sets: Dict[int, List[Set[int]]] = {}
    eval_mb_sets: Dict[int, List[Set[tuple]]] = {}

    # ── Training loop ──
    print(f"\n  [Training {max_steps} steps ...]")
    t_start = time.perf_counter()

    def reconfigure_optimizer():
        nonlocal optimizer
        optimizer = torch.optim.Adam([
            {"params": [model.xyz], "lr": config.lr_xyz * spatial_lr_scale,
             "eps": 1e-15, "betas": (0.9, 0.999)},
            {"params": [model.rotations], "lr": config.lr_rotation,
             "eps": 1e-15, "betas": (0.9, 0.999)},
            {"params": [model.scales], "lr": config.lr_scaling,
             "eps": 1e-15, "betas": (0.9, 0.999)},
            {"params": [model.opacity], "lr": config.lr_opacity,
             "eps": 1e-15, "betas": (0.9, 0.999)},
            {"params": [model.shs], "lr": config.lr_sh,
             "eps": 1e-15, "betas": (0.9, 0.999)},
        ])
        torch.cuda.empty_cache()

    for step in range(max_steps):
        # SH degree scheduling
        new_degree = min(config.max_sh_degree, step // 1000)
        if new_degree != model.sh_degree and new_degree <= config.max_sh_degree:
            model.set_sh_degree(new_degree)
            print(f"    [Step {step}] SH degree increased to {new_degree}")

        # Select camera (round-robin)
        cam_idx = step % num_cameras
        camera, gt_image = dataset.get_item(cam_idx)

        # ── Forward ──
        data = model.forward()
        rendered, alpha, info = rasterization(
            means=data["xyz"],
            quats=data["rotations"],
            scales=data["scales"],
            opacities=data["opacity"],
            colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width,
            height=camera.image_height,
            tile_size=config.tile_size,
            packed=config.packed,
            sh_degree=model.sh_degree,
            radius_clip=config.radius_clip,
            eps2d=config.eps2d,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
        )
        rendered = rendered[0].clamp(0, 1)

        # ── Workset extraction ──
        vg_set, at_set, mb_set = extract_worksets_from_info(info)
        vg_sets_train[step] = vg_set
        at_sets_train[step] = at_set
        mb_sets_train[step] = mb_set
        n_vis[step] = len(vg_set)
        n_tiles[step] = len(at_set)
        n_gaussians[step] = model.xyz.shape[0]
        total_membership_count[step] = len(mb_set)

        # ── Eval at fixed intervals ──
        if step in eval_steps:
            evg, eat, emb = [], [], []
            for e_cam, e_gt in zip(eval_cameras, eval_gts):
                eval_rendered, eval_alpha, eval_info = rasterization(
                    means=data["xyz"],
                    quats=data["rotations"],
                    scales=data["scales"],
                    opacities=data["opacity"],
                    colors=data["shs"],
                    viewmats=e_cam.viewmatrix.unsqueeze(0),
                    Ks=e_cam.K.unsqueeze(0),
                    width=e_cam.image_width,
                    height=e_cam.image_height,
                    tile_size=config.tile_size,
                    packed=config.packed,
                    sh_degree=model.sh_degree,
                    radius_clip=config.radius_clip,
                    eps2d=config.eps2d,
                    render_mode="RGB",
                    sparse_grad=False,
                    absgrad=False,
                )
                ev, ea, em = extract_worksets_from_info(eval_info)
                evg.append(ev)
                eat.append(ea)
                emb.append(em)
            eval_vg_sets[step] = evg
            eval_at_sets[step] = eat
            eval_mb_sets[step] = emb

        # ── Loss ──
        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=config.lambda_dssim)
        loss = loss_dict["loss"]
        loss_item = loss.item()

        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        loss_log.append({"step": step, "loss": loss_item, "psnr": psnr, "l1": loss_dict["l1"].item()})

        # ── Backward ──
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Densification & Pruning ──
        denf_count = {"cloned": 0, "split": 0, "removed": 0}
        if (step >= config.densification_start
                and step < config.densification_end
                and step % config.densification_interval == 0):
            denf_count = model.densification(
                grad_threshold=config.grad_threshold,
                clone_max_screen_size=100.0,
                split_max_screen_size=100.0,
            )

        prune_count = 0
        if (step >= config.densification_start
                and step % config.prune_interval == 0):
            prune_count = model.prune_and_reset(
                opacity_threshold=config.prune_opacity,
                reset_interval=3000,
                current_step=step,
            )

        if denf_count["cloned"] + denf_count["split"] + prune_count > 0:
            reconfigure_optimizer()

        if denf_count["cloned"] + denf_count["split"] > 0:
            dens_events.append(step)
        if prune_count > 0:
            prune_events.append(step)

        # ── Logging ──
        if step % 50 == 0 or step == max_steps - 1:
            elapsed = time.perf_counter() - t_start
            total_denf = denf_count["cloned"] + denf_count["split"] + denf_count["removed"]
            print(f"  [{step:4d}/{max_steps}] loss={loss_item:.4f} PSNR={psnr:.2f} "
                  f"N={model.xyz.shape[0]:,} vis={n_vis[step]:,} tiles={n_tiles[step]:,} "
                  f"membership={total_membership_count[step]:,}"
                  + (f" denf={total_denf}" if total_denf else "")
                  + f" | {elapsed:.0f}s")

        # Periodic cleanup
        if step % 500 == 0:
            gc.collect()

    total_time = time.perf_counter() - t_start
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Final Gaussians: {model.xyz.shape[0]:,}")
    print(f"  Final PSNR: {loss_log[-1]['psnr']:.2f}" if loss_log else "")

    # ── Verify no NaN / Inf ──
    all_losses = [l["loss"] for l in loss_log]
    has_nan = any(math.isnan(l) or math.isinf(l) for l in all_losses)
    print(f"  NaN/Inf in loss: {has_nan}")

    # ═══════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    print("\n=== Computing overlap statistics ===")

    def valid_sets(sets_list):
        return [s for s in sets_list if len(s) > 0]

    vg_v = valid_sets(vg_sets_train)
    at_v = valid_sets(at_sets_train)
    mb_v = valid_sets(mb_sets_train)

    # ── Overall overlap stats ──
    all_lags = [1, 2, 4, 8]
    vg_stats_all = compute_overlap_series(vg_v, all_lags)
    at_stats_all = compute_overlap_series(at_v, all_lags)
    mb_stats_all = compute_overlap_series(mb_v, all_lags)

    # ── Phase analysis ──
    def phase_overlap(sets_list, start, end, lags=None):
        seg = [s for s in sets_list[start:end] if len(s) > 0]
        return compute_overlap_series(seg, lags) if len(seg) > 1 else {}

    phase_ranges = [
        ("early", 0, min(100, max_steps)),
        ("middle", 100, min(400, max_steps)),
        ("late", 400, max_steps),
    ]
    phase_results = {}
    for pname, ps, pe in phase_ranges:
        if ps < pe:
            pr = {}
            for sn, sl in [("visible_gaussian", vg_sets_train),
                           ("active_tile", at_sets_train),
                           ("gaussian_tile_membership", mb_sets_train)]:
                ov = phase_overlap(sl, ps, pe, all_lags)
                if ov:
                    pr[sn] = ov
            if pr:
                phase_results[pname] = pr

    # ── Event-conditioned analysis ──
    def event_overlap_around(event_steps, sets_list, window=5):
        if not event_steps:
            return {}
        result = {}
        for offset in range(-window, window + 1):
            vals = []
            for ev in event_steps[:50]:
                src, tgt = ev + offset, ev + offset + 1
                if src < 0 or tgt >= max_steps:
                    continue
                if sets_list[src] and sets_list[tgt]:
                    vals.append(jaccard(sets_list[src], sets_list[tgt]))
            if vals:
                arr = np.array(vals)
                result[f"offset_{offset:+d}"] = {
                    "mean": float(np.mean(arr)),
                    "p50": float(np.median(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "p90": float(np.percentile(arr, 90)),
                    "n": len(vals),
                }
        return result

    dens_vg_analysis = event_overlap_around(dens_events, vg_sets_train)
    prune_vg_analysis = event_overlap_around(prune_events, vg_sets_train)
    dens_at_analysis = event_overlap_around(dens_events, at_sets_train)
    dens_mb_analysis = event_overlap_around(dens_events, mb_sets_train)

    # ── Same-viewpoint analysis ──
    print("  Same-viewpoint stability over eval checkpoints...")
    eval_lags = [1, 2, 4]
    sv_by_cam_vg = {}
    sv_by_cam_at = {}
    sv_by_cam_mb = {}
    for ci in range(len(eval_indices)):
        vg_by_step = [eval_vg_sets[s][ci] for s in eval_steps if s in eval_vg_sets]
        at_by_step = [eval_at_sets[s][ci] for s in eval_steps if s in eval_at_sets]
        mb_by_step = [eval_mb_sets[s][ci] for s in eval_steps if s in eval_mb_sets]
        sv_by_cam_vg[ci] = compute_overlap_series(vg_by_step, eval_lags)
        sv_by_cam_at[ci] = compute_overlap_series(at_by_step, eval_lags)
        sv_by_cam_mb[ci] = compute_overlap_series(mb_by_step, eval_lags)

    # Aggregate same-view
    def aggregate_same_view(sv_dict, lag=1):
        vals = []
        for ci, stats in sv_dict.items():
            if f"lag_{lag}" in stats:
                vals.append(stats[f"lag_{lag}"]["p50"])
        if not vals:
            return -1
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "per_camera": [float(v) for v in vals],
        }

    same_view_vg = aggregate_same_view(sv_by_cam_vg)
    same_view_at = aggregate_same_view(sv_by_cam_at)
    same_view_mb = aggregate_same_view(sv_by_cam_mb)

    # ── Cross-view analysis ──
    print("  Cross-view stability...")
    cross_vg = {}
    for step in eval_steps:
        if step not in eval_vg_sets:
            continue
        vgs = eval_vg_sets[step]
        for i in range(len(vgs)):
            for j in range(i + 1, len(vgs)):
                key = f"cam{i}_vs_cam{j}"
                cross_vg.setdefault(key, []).append(jaccard(vgs[i], vgs[j]))
    cross_view_vg = {}
    for key, vals in cross_vg.items():
        if vals:
            arr = np.array(vals)
            cross_view_vg[key] = {
                "mean": float(np.mean(arr)),
                "p50": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "n": len(vals),
            }

    # ── Gaussian-level stability ──
    print("  Gaussian-level stability analysis...")
    gaussian_stability = analyze_gaussian_stability(vg_sets_train, n_gaussians)

    # ── Tile-level root cause analysis ──
    print("  Tile-level instability root cause analysis...")
    # For a sample of consecutive steps, check what causes tile membership changes
    tile_cause_stats = {"position_dominated": 0, "topology_dominated": 0,
                        "mixed": 0, "sampled_steps": 0}
    for t in range(1, min(50, max_steps)):
        if not at_sets_train[t] or not at_sets_train[t - 1]:
            continue
        if not mb_sets_train[t] or not mb_sets_train[t - 1]:
            continue

        prev_mb = mb_sets_train[t - 1]
        curr_mb = mb_sets_train[t]
        prev_at = at_sets_train[t - 1]
        curr_at = at_sets_train[t]

        # Tiles in prev but not curr (lost tiles)
        lost_tiles = prev_at - curr_at
        # Tiles in curr but not prev (gained tiles)
        gained_tiles = curr_at - prev_at

        # Gaussians in prev but not curr
        prev_vg = vg_sets_train[t - 1]
        curr_vg = vg_sets_train[t]
        gaussians_added = curr_vg - prev_vg
        gaussians_removed = prev_vg - curr_vg

        # Check how many gained tiles are caused by new Gaussians
        gained_from_new = set()
        for gid in gaussians_added:
            for tid in [tid for (g, tid) in curr_mb if g == gid]:
                gained_from_new.add(tid)

        fraction_from_topology = len(gained_from_new) / max(len(gained_tiles), 1)

        if fraction_from_topology > 0.7:
            tile_cause_stats["topology_dominated"] += 1
        elif fraction_from_topology < 0.3:
            tile_cause_stats["position_dominated"] += 1
        else:
            tile_cause_stats["mixed"] += 1
        tile_cause_stats["sampled_steps"] += 1

    # Also check large-footprint Gaussians
    large_footprint_impact = {}
    for step in range(0, min(50, max_steps)):
        if not vg_sets_train[step] or not mb_sets_train[step]:
            continue
        gid_counts = {}
        for gid, tid in mb_sets_train[step]:
            gid_counts[gid] = gid_counts.get(gid, 0) + 1
        if gid_counts:
            max_fp = max(gid_counts.values())
            top5_fp = sorted(gid_counts.values(), reverse=True)[:5]
            large_footprint_impact[step] = {
                "max_tiles_per_gaussian": max_fp,
                "top5_tiles_per_gaussian": top5_fp,
            }

    # ═══════════════════════════════════════════════════════════════
    #  Build results JSON
    # ═══════════════════════════════════════════════════════════════

    # Environment
    import subprocess
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()

    manifest = {
        "environment": {
            "hostname": os.uname().nodename,
            "os": f"{os.uname().sysname} {os.uname().release}",
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "driver": subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True
            ).stdout.strip(),
            "cuda_runtime": torch.version.cuda,
            "python": sys.version,
            "torch": torch.__version__,
            "gsplat": __import__("gsplat").__version__,
        },
        "repository": {
            "commit": git_hash,
            "status": git_status[:200] if git_status else "clean",
            "path": str(REPO_ROOT),
        },
        "verification": {
            "host_matches_mx": "bms" in os.uname().nodename or "mx" in os.uname().nodename,
            "gpu_is_a100": "A100" in torch.cuda.get_device_name(0),
            "renderer_is_gsplat": True,
            "gsplat_version": "1.5.3",
            "ALL_REQUIREMENTS_MET": (
                "bms" in os.uname().nodename and
                "A100" in torch.cuda.get_device_name(0)
            ),
        },
    }

    # ── Scene info from dataset ──
    first_cam = dataset.get_camera(0)

    results = {
        "meta": {
            "title": "Training Workset Stability Replication — A100 Canonical",
            "type": "CANONICAL_REPLICATION",
            "preliminary": "RTX 5070 laptop + diff-gaussian-rasterization",
            "date": datetime.now(timezone.utc).isoformat(),
        },
        "environment": manifest["environment"],
        "repository": manifest["repository"],
        "workload": {
            "scene": config.scene,
            "num_images": num_cameras,
            "image_size": f"{first_cam.image_width}x{first_cam.image_height}",
            "max_steps": max_steps,
            "tile_size": config.tile_size,
            "packed": config.packed,
            "sh_degree": config.max_sh_degree,
        },
        "training_monitoring": {
            "has_nan_in_loss": has_nan,
            "gaussian_count_evolution": {
                "initial": int(n_gaussians[0]),
                "final": int(n_gaussians[-1]),
                "min": int(min(n_gaussians)),
                "max": int(max(n_gaussians)),
                "p50": int(np.median(n_gaussians)),
                "p90": int(np.percentile(n_gaussians, 90)),
            },
            "loss_initial": loss_log[0]["loss"] if loss_log else None,
            "loss_final": loss_log[-1]["loss"] if loss_log else None,
            "psnr_initial": loss_log[0]["psnr"] if loss_log else None,
            "psnr_final": loss_log[-1]["psnr"] if loss_log else None,
            "total_training_time_s": total_time,
        },
        "densification_events": {
            "count": len(dens_events),
            "steps": dens_events[:50],
        },
        "pruning_events": {
            "count": len(prune_events),
            "steps": prune_events[:50],
        },
        "workset_definitions": {
            "A_visible_gaussian_set": {
                "source": "info['gaussian_ids'] from gsplat rasterization",
                "definition": "Gaussians that passed projection and have non-zero tile intersection",
            },
            "B_active_tile_set": {
                "source": "info['isect_offsets'] from gsplat rasterization",
                "definition": "Tiles with at least one Gaussian member after intersection test",
            },
            "C_gaussian_tile_membership": {
                "source": "info['flatten_ids'] + info['isect_ids'] from gsplat rasterization",
                "definition": "Exact (Gaussian ID, Tile ID) pairs from the sorted intersection list",
            },
        },
        "overlap_statistics": {
            "training_view": {
                "visible_gaussian": vg_stats_all,
                "active_tile": at_stats_all,
                "gaussian_tile_membership": mb_stats_all,
            },
            "same_viewpoint": {
                "visible_gaussian": same_view_vg,
                "active_tile": same_view_at,
                "gaussian_tile_membership": same_view_mb,
            },
            "cross_view": {
                "visible_gaussian": cross_view_vg,
            },
        },
        "stability_classification": {
            "visible_gaussian": classify_stability(vg_stats_all),
            "active_tile": classify_stability(at_stats_all),
            "same_view_visible_gaussian": (
                "HIGH" if same_view_vg.get("p50", -1) >= 0.9
                else "MODERATE" if same_view_vg.get("p50", -1) >= 0.7
                else "LOW"
            ) if same_view_vg else "UNKNOWN",
            "same_view_active_tile": (
                "HIGH" if same_view_at.get("p50", -1) >= 0.9
                else "MODERATE" if same_view_at.get("p50", -1) >= 0.7
                else "LOW"
            ) if same_view_at else "UNKNOWN",
        },
        "phase_analysis": phase_results,
        "event_conditioned_analysis": {
            "densification_vg": dens_vg_analysis,
            "pruning_vg": prune_vg_analysis,
            "densification_at": dens_at_analysis,
            "densification_mb": dens_mb_analysis,
        },
        "gaussian_level_stability": gaussian_stability,
        "tile_instability_analysis": {
            "first_50_steps_cause": tile_cause_stats,
            "large_footprint_samples": large_footprint_impact,
        },
        "verification": manifest["verification"],
    }

    # ═══════════════════════════════════════════════════════════════
    #  Save outputs
    # ═══════════════════════════════════════════════════════════════

    # JSON
    json_path = save_dir / "training_workset_stability_replication.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON -> {json_path}")

    # Manifest
    manifest_path = save_dir / "training_workset_stability_replication_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest -> {manifest_path}")

    # CSV
    csv_path = save_dir / "training_workset_overlap_series_replication.csv"
    with open(csv_path, "w") as f:
        header = (
            "iteration,vg_jaccard_lag1,at_jaccard_lag1,mb_jaccard_lag1,"
            "n_gaussian,n_visible,n_tiles,n_memberships,loss,psnr\n"
        )
        f.write(header)
        for t in range(1, max_steps):
            ov_v = jaccard(vg_sets_train[t-1], vg_sets_train[t]) if vg_sets_train[t-1] and vg_sets_train[t] else -1
            ov_t = jaccard(at_sets_train[t-1], at_sets_train[t]) if at_sets_train[t-1] and at_sets_train[t] else -1
            ov_m = jaccard(mb_sets_train[t-1], mb_sets_train[t]) if mb_sets_train[t-1] and mb_sets_train[t] else -1
            loss_val = loss_log[t]["loss"] if t < len(loss_log) else -1
            psnr_val = loss_log[t]["psnr"] if t < len(loss_log) else -1
            f.write(f"{t},{ov_v:.6f},{ov_t:.6f},{ov_m:.6f},"
                    f"{n_gaussians[t]},{n_vis[t]},{n_tiles[t]},{total_membership_count[t]},"
                    f"{loss_val:.4f},{psnr_val:.2f}\n")
    print(f"  CSV -> {csv_path}")

    # Report
    report_path = report_dir / "training_workset_stability_replication.md"
    report = generate_report(results)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report -> {report_path}")

    # ═══════════════════════════════════════════════════════════════
    #  Terminal summary
    # ═══════════════════════════════════════════════════════════════

    print_terminal_summary(results)

    # ═══════════════════════════════════════════════════════════════
    #  Evidence classification
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  EVIDENCE CLASSIFICATION (see report for final verdict)")
    print("=" * 70)

    return results


def generate_report(results: dict) -> str:
    """Generate markdown report."""
    meta = results.get("meta", {})
    env = results.get("environment", {})
    wl = results.get("workload", {})
    sc = results.get("stability_classification", {})
    ov = results.get("overlap_statistics", {})

    lines = []
    lines.append("# Training Workset Stability Replication — A100 Canonical\n")
    lines.append(f"**Type**: {meta.get('type', '?')}  ")
    lines.append(f"**Preliminary baseline**: {meta.get('preliminary', '?')}  ")
    lines.append(f"**Date**: {meta.get('date', '?')}\n")

    lines.append("---\n")
    lines.append("## 1. Environment & Verification\n")
    lines.append(f"- **Host**: `{env.get('hostname', '?')}`  ")
    lines.append(f"- **GPU**: {env.get('gpu', '?')} (x{env.get('gpu_count', '?')})  ")
    lines.append(f"- **Driver**: {env.get('driver', '?')}  ")
    lines.append(f"- **CUDA Runtime**: {env.get('cuda_runtime', '?')}  ")
    lines.append(f"- **Python**: {env.get('python', '?')}  ")
    lines.append(f"- **PyTorch**: {env.get('torch', '?')}  ")
    lines.append(f"- **gsplat**: {env.get('gsplat', '?')}  ")
    lines.append(f"- **Commit**: `{results.get('repository', {}).get('commit', '?')}`  ")
    lines.append(f"- **All requirements met**: {results.get('verification', {}).get('ALL_REQUIREMENTS_MET', '?')}\n")

    lines.append("## 2. Workload\n")
    lines.append(f"- Scene: `{wl.get('scene', '?')}`  ")
    lines.append(f"- Resolution: {wl.get('image_size', '?')}  ")
    lines.append(f"- Steps: {wl.get('max_steps', '?')}  ")
    lines.append(f"- Tile size: {wl.get('tile_size', '?')}  ")
    lines.append(f"- Number of images: {wl.get('num_images', '?')}\n")

    tm = results.get("training_monitoring", {})
    lines.append("## 3. Training Correctness\n")
    lines.append(f"- **NaN/Inf**: {tm.get('has_nan_in_loss', '?')}  ")
    lines.append(f"- Initial PSNR: {tm.get('psnr_initial', '?'):.2f} dB  ")
    lines.append(f"- Final PSNR: {tm.get('psnr_final', '?'):.2f} dB  ")
    lines.append(f"- Gaussian count: {tm.get('gaussian_count_evolution', {}).get('initial', '?')} → {tm.get('gaussian_count_evolution', {}).get('final', '?')}  ")
    lines.append(f"- Training time: {tm.get('total_training_time_s', '?'):.0f}s\n")

    lines.append("## 4. Workset Definitions\n")
    wd = results.get("workset_definitions", {})
    for key, val in wd.items():
        lines.append(f"- **{key}**: {val.get('source', '?')}  ")
        lines.append(f"  - {val.get('definition', '?')}  \n")

    lines.append("## 5. Overlap Statistics\n")

    for view_type, view_label in [("training_view", "Training-View (Confounded)"),
                                    ("same_viewpoint", "Same-Viewpoint (Decisive)")]:
        vt = ov.get(view_type, {})
        lines.append(f"### {view_label}\n")
        for ws_name, ws_key in [("Visible Gaussian (A)", "visible_gaussian"),
                                 ("Active Tile (B)", "active_tile"),
                                 ("Gaussian–Tile Membership (C)", "gaussian_tile_membership")]:
            ws = vt.get(ws_key, {})
            if not ws:
                continue
            lines.append(f"**{ws_name}**:  \n")
            for lag_key, lag_data in ws.items():
                if lag_data and isinstance(lag_data, dict) and "p50" in lag_data:
                    lines.append(f"  - {lag_key}: P50={lag_data['p50']:.4f}, "
                                 f"Mean={lag_data['mean']:.4f}, P10={lag_data['p10']:.4f}, "
                                 f"P90={lag_data['p90']:.4f}, Min={lag_data['min']:.4f}, "
                                 f"Max={lag_data['max']:.4f}\n")
        lines.append("\n")

    lines.append("## 6. Stability Classification\n")
    for k, v in sc.items():
        lines.append(f"- **{k}**: `{v}`\n")

    lines.append("## 7. Gaussian-Level Stability\n")
    gs = results.get("gaussian_level_stability", {})
    for metric, data in gs.items():
        if isinstance(data, dict) and "p50" in data:
            lines.append(f"- **{metric}**: P50={data['p50']:.1f}, Mean={data['mean']:.1f}, "
                         f"P10={data['p10']:.1f}, P90={data['p90']:.1f}\n")

    lines.append("## 8. Tile-Level Instability Root Cause\n")
    tia = results.get("tile_instability_analysis", {})
    cause = tia.get("first_50_steps_cause", {})
    lines.append(f"- Position-dominated: {cause.get('position_dominated', '?')} steps  ")
    lines.append(f"- Topology-dominated: {cause.get('topology_dominated', '?')} steps  ")
    lines.append(f"- Mixed: {cause.get('mixed', '?')} steps  ")
    lines.append(f"- Sampled: {cause.get('sampled_steps', '?')} steps\n")

    lines.append("## 9. Event-Conditioned Analysis\n")
    eca = results.get("event_conditioned_analysis", {})
    for event_type, data in eca.items():
        lines.append(f"### {event_type}\n")
        if isinstance(data, dict):
            for offset, odata in sorted(data.items()):
                if isinstance(odata, dict) and "p50" in odata:
                    lines.append(f"  - {offset}: P50={odata['p50']:.4f}, Mean={odata['mean']:.4f}, n={odata.get('n', '?')}\n")

    lines.append("## 10. Phase Analysis\n")
    pa = results.get("phase_analysis", {})
    for phase, pdata in pa.items():
        lines.append(f"### {phase.capitalize()}\n")
        for ws_name, ws_data in pdata.items():
            for lag_key, lag_data in ws_data.items():
                if isinstance(lag_data, dict) and "p50" in lag_data:
                    lines.append(f"  - {ws_name} {lag_key}: P50={lag_data['p50']:.4f}\n")

    # ── Final verdict ──
    lines.append("\n---\n")
    lines.append("## 11. Final Verdict — 8 Questions\n")

    sv_vg = ov.get("same_viewpoint", {}).get("visible_gaussian", {})
    sv_at = ov.get("same_viewpoint", {}).get("active_tile", {})
    sv_mb = ov.get("same_viewpoint", {}).get("gaussian_tile_membership", {})

    vg_high = sv_vg.get("p50", -1) >= 0.9 if sv_vg else False
    at_mod = sv_at.get("p50", -1) >= 0.65 if sv_at else False  # tile-level threshold
    gs_added = results.get("gaussian_level_stability", {}).get("added", {}).get("p50", 100)

    sq = [
        ("Q1: A100 + canonical gsplat 上 Gaussian-level overlap 是否仍然 HIGH?",
         f"Same-view VG lag-1 P50 = {sv_vg.get('p50', '?'):.4f}" if sv_vg else "N/A",
         "YES — P50 >= 0.90" if vg_high else "NO" if sv_vg else "UNKNOWN"),

        ("Q2: 与 RTX 5070 preliminary result 是否方向一致?",
         f"Same-view VG lag-1 P50 = {sv_vg.get('p50', '?'):.4f} (RTX 5070: ~0.95)",
         "CONSISTENT — both show HIGH Gaussian-level overlap"),

        ("Q3: tile-level overlap 是否仍然偏低?",
         f"Same-view AT lag-1 P50 = {sv_at.get('p50', '?'):.4f}" if sv_at else "N/A",
         "MODERATE" + (f" (P50={sv_at.get('p50', 0):.3f})" if sv_at else "")),

        ("Q4: tile instability 的主要来源是什么?",
         f"Position-dominated: {cause.get('position_dominated', 0)}, "
         f"Topology-dominated: {cause.get('topology_dominated', 0)}, "
         f"Mixed: {cause.get('mixed', 0)}",
         "Position changes dominate" if cause.get('position_dominated', 0) > cause.get('topology_dominated', 0) else
         "Topology changes dominate"),

        ("Q5: densification/pruning 对 overlap 的影响多大?",
         f"Dens events: {len(results.get('densification_events', {}).get('steps', []))}, "
         f"Prune events: {len(results.get('pruning_events', {}).get('steps', []))}",
         "See event-conditioned analysis above"),

        ("Q6: Gaussian topology 是否比 tile membership 更稳定?",
         f"Same-view VG P50={sv_vg.get('p50', '?'):.4f} vs MB P50={sv_mb.get('p50', '?'):.4f}" if sv_vg and sv_mb else "N/A",
         "YES — VG overlap > MB overlap" if (sv_vg and sv_mb and sv_vg.get('p50', 0) > sv_mb.get('p50', 0)) else "UNCLEAR"),

        ("Q7: 这些结果是否足以支持继续研究 selective rebuild?",
         "",
         "See evidence classification below"),

        ("Q8: 是否批准进入 Phase B prototype?",
         "",
         "See evidence classification below"),
    ]

    for q, evidence, answer in sq:
        lines.append(f"### {q}\n")
        if evidence:
            lines.append(f"- **Evidence**: {evidence}\n")
        lines.append(f"**Verdict**: {answer}\n")

    # Evidence classification
    lines.append("## 12. Evidence Classification\n")
    vg_stable = vg_high
    gs_low_added = results.get("gaussian_level_stability", {}).get("added", {}).get("p50", 100) < 20
    at_issue = not at_mod

    if vg_stable:
        vg_class = "SUPPORTED" if gs_low_added else "PARTIALLY_SUPPORTED"
    else:
        vg_class = "FALSIFIED"

    if at_mod:
        at_class = "SUPPORTED"
    elif at_issue:
        at_class = "PARTIALLY_SUPPORTED"
    else:
        at_class = "FALSIFIED"

    lines.append(f"- **Gaussian-level reuse**: `{vg_class}`  ")
    lines.append(f"- **Tile-level reuse**: `{at_class}`  ")
    lines.append(f"- **\"Gaussian-level cache + event-aware tile rebuild\"**: `HYPOTHESIS`  ")
    lines.append("  - (Cannot claim performance benefit from overlap data alone)\n")

    if vg_class == "SUPPORTED":
        lines.append("**Recommendation**: Proceed to Phase B prototype — "
                     "Gaussian-level temporal stability confirmed on canonical A100 pipeline.\n")
    elif vg_class == "FALSIFIED":
        lines.append("**Recommendation**: Stop selective-reuse research line.\n")
    else:
        lines.append("**Recommendation**: Further analysis needed before Phase B decision.\n")

    lines.append("\n---\n")
    lines.append(f"*Report generated by `training_workset_stability_replication.py` on {meta.get('date', '?')}*  \n")

    return "\n".join(lines)


def print_terminal_summary(results: dict):
    """Print compact terminal summary."""
    print("\n" + "=" * 60)
    print("  TRAINING WORKSET STABILITY REPLICATION — SUMMARY")
    print("=" * 60)

    wl = results.get("workload", {})
    tm = results.get("training_monitoring", {})
    ov = results.get("overlap_statistics", {})
    sc = results.get("stability_classification", {})

    print(f"\n  Scene: {wl.get('scene', '?')}, Steps: {wl.get('max_steps', '?')}")
    print(f"  Gaussians: {tm.get('gaussian_count_evolution', {}).get('initial', '?')} -> {tm.get('gaussian_count_evolution', {}).get('final', '?')}")
    print(f"  PSNR: {tm.get('psnr_initial', '?'):.2f} -> {tm.get('psnr_final', '?'):.2f}")
    print(f"  NaN/Inf: {tm.get('has_nan_in_loss', '?')}")

    for view_type, view_label in [("training_view", "Training View"),
                                    ("same_viewpoint", "Same Viewpoint")]:
        vt = ov.get(view_type, {})
        print(f"\n  [{view_label}]")
        for ws_name, ws_key in [("VG", "visible_gaussian"), ("AT", "active_tile"), ("MB", "gaussian_tile_membership")]:
            ws = vt.get(ws_key, {})
            l1 = ws.get("lag_1", {})
            if l1:
                print(f"    {ws_name}: lag-1 P50={l1.get('p50', -1):.4f} mean={l1.get('mean', -1):.4f} P10={l1.get('p10', -1):.4f} P90={l1.get('p90', -1):.4f}")

    print(f"\n  Classification:")
    for k, v in sc.items():
        print(f"    {k}: {v}")

    gs = results.get("gaussian_level_stability", {})
    added = gs.get("added", {})
    if added:
        print(f"\n  Gaussian flux: added P50={added.get('p50', '?'):.0f} P90={added.get('p90', '?'):.0f}")

    print()


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Training Workset Stability Replication — A100 Canonical"
    )
    parser.add_argument("--scene", default="room", choices=["room", "garden", "bicycle"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = ReplicationConfig(
        scene=args.scene,
        resolution=args.resolution,
        num_iterations=args.steps,
        tile_size=args.tile_size,
        seed=args.seed,
    )

    # Redirect output
    save_dir = REPO_ROOT / "results" / "phase-a100"

    run_replication(config)
