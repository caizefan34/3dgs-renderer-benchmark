#!/usr/bin/env python3
"""
C38: Training-Time Projection/Intersection Reuse Validity Study.

Exact validity test: does state_(t) == state_(t+k) for k ∈ {1,2,4,8,16}?

Key design:
- packed=True for training (as specified)
- Capture gaussian_ids to remap means2d to global indices for comparison
- Exact equality test on INTERSECTION state (flatten_ids, isect_offsets)
- Continuous delta measurement on PROJECTION state (means2d, conics, depths)
- No densification/pruning (topology fixed)
- 500 iterations, fixed camera
"""
import json, math, os, time, sys
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from loss import combined_loss
from dataset import GTDataset


@dataclass
class StateSnapshot:
    """Exact snapshot of renderer state at one iteration."""
    iter: int
    # Packed projection state (only visible GS)
    means2d_packed: torch.Tensor      # [nnz, 2] f32
    conics_packed: torch.Tensor       # [nnz, 3] f32
    depths_packed: torch.Tensor       # [nnz] f32
    radii_packed: torch.Tensor        # [nnz] f32
    gaussian_ids: torch.Tensor        # [nnz] int32 — maps packed idx → global GS idx
    # Intersection state
    flatten_ids: torch.Tensor         # [n_isects] int32
    isect_offsets: torch.Tensor       # [C, tile_h, tile_w] int32
    # Metadata
    n_gaussians: int
    n_visible: int
    n_intersections: int
    tile_width: int
    tile_height: int


@torch.no_grad()
def capture_state(model, camera, config, sh_degree, iteration) -> StateSnapshot:
    """Run forward and capture all renderer state from meta dict."""
    data = model.forward()
    _, _, info = rasterization(
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
        sh_degree=sh_degree,
        radius_clip=config.radius_clip,
        eps2d=config.eps2d,
        render_mode="RGB",
        sparse_grad=False,
        absgrad=False,
    )
    # Handle radii shape: gsplat 1.5.3 may return [nnz, 2] int32 in packed mode
    radii_data = info["radii"].detach().cpu().clone()
    if radii_data.dim() > 1:
        radii_data = radii_data.max(dim=-1).values  # flatten to 1D
    radii_data = radii_data.float()  # ensure float dtype for comparisons

    # Handle depths shape: may be [nnz, 1] in some versions
    depths_data = info["depths"].detach().cpu().clone()
    if depths_data.dim() > 1:
        depths_data = depths_data.squeeze(-1)
    depths_data = depths_data.float()

    # Handle conics shape: may be [nnz, 3] or [nnz, 4]
    conics_data = info["conics"].detach().cpu().clone()
    if conics_data.dim() > 1 and conics_data.shape[-1] > 3:
        conics_data = conics_data[..., :3]  # take first 3 columns
    conics_data = conics_data.float()

    return StateSnapshot(
        iter=iteration,
        means2d_packed=info["means2d"].detach().cpu().clone(),
        conics_packed=conics_data,
        depths_packed=depths_data,
        radii_packed=radii_data,
        gaussian_ids=info["gaussian_ids"].detach().cpu().clone(),
        flatten_ids=info["flatten_ids"].detach().cpu().clone(),
        isect_offsets=info["isect_offsets"].detach().cpu().clone(),
        n_gaussians=data["xyz"].shape[0],
        n_visible=info["means2d"].shape[0],
        n_intersections=info["flatten_ids"].shape[0],
        tile_width=info["tile_width"],
        tile_height=info["tile_height"],
    )


def compare_states(sa: StateSnapshot, sb: StateSnapshot):
    """
    Compare two state snapshots.
    Returns dict with:
    - means2d deltas (continuous, for GS visible in BOTH)
    - radii exact equality
    - flatten_ids exact equality
    - isect_offsets exact equality
    - intersection exact equality (flatten_ids AND isect_offsets)
    """
    result = {}

    # --- Projection state (continuous) ---
    # Remap packed means2d to global indices using gaussian_ids
    N = sa.n_gaussians  # same N (no densification)
    gids_a = sa.gaussian_ids  # [nnz_a]
    gids_b = sb.gaussian_ids  # [nnz_b]

    # Create global means2d arrays
    m2d_a = torch.zeros(N, 2, dtype=torch.float32)
    m2d_b = torch.zeros(N, 2, dtype=torch.float32)
    radii_a = torch.zeros(N, dtype=torch.float32)
    radii_b = torch.zeros(N, dtype=torch.float32)
    depths_a = torch.zeros(N, dtype=torch.float32)
    depths_b = torch.zeros(N, dtype=torch.float32)
    # Dynamic conics dimension
    conics_dim = sa.conics_packed.shape[-1] if sa.conics_packed.dim() > 1 else 1
    conics_a = torch.zeros(N, conics_dim, dtype=torch.float32)
    conics_b = torch.zeros(N, conics_dim, dtype=torch.float32)

    m2d_a[gids_a] = sa.means2d_packed
    m2d_b[gids_b] = sb.means2d_packed
    radii_a[gids_a] = sa.radii_packed
    radii_b[gids_b] = sb.radii_packed
    depths_a[gids_a] = sa.depths_packed
    depths_b[gids_b] = sb.depths_packed
    conics_a[gids_a] = sa.conics_packed
    conics_b[gids_b] = sb.conics_packed

    # Compare only GS visible in BOTH
    vis_both = (radii_a > 0) & (radii_b > 0)
    n_vis_both = vis_both.sum().item()
    result["n_vis_both"] = n_vis_both

    if n_vis_both > 0:
        m2d_diff = (m2d_a[vis_both] - m2d_b[vis_both]).abs()
        result["means2d_max_delta"] = m2d_diff.max().item()
        result["means2d_mean_delta"] = m2d_diff.mean().item()

        depth_diff = (depths_a[vis_both] - depths_b[vis_both]).abs()
        result["depths_max_delta"] = depth_diff.max().item()

        conics_diff = (conics_a[vis_both] - conics_b[vis_both]).abs()
        result["conics_max_delta"] = conics_diff.max().item()
    else:
        result["means2d_max_delta"] = float("inf")
        result["means2d_mean_delta"] = float("inf")
        result["depths_max_delta"] = float("inf")
        result["conics_max_delta"] = float("inf")

    # Visibility changes
    vis_a_only = (radii_a > 0) & (radii_b == 0)
    vis_b_only = (radii_a == 0) & (radii_b > 0)
    result["n_became_visible"] = vis_b_only.sum().item()
    result["n_became_invisible"] = vis_a_only.sum().item()
    result["n_visibility_changed"] = result["n_became_visible"] + result["n_became_invisible"]

    # --- Intersection state (discrete/exact) ---
    # n_isects
    result["n_isects_a"] = sa.n_intersections
    result["n_isects_b"] = sb.n_intersections
    result["n_isects_changed"] = sb.n_intersections - sa.n_intersections

    # flatten_ids exact equality
    if sa.flatten_ids.shape == sb.flatten_ids.shape:
        result["flatten_ids_exact_eq"] = sa.flatten_ids.equal(sb.flatten_ids)
    else:
        result["flatten_ids_exact_eq"] = False

    # isect_offsets exact equality
    if sa.isect_offsets.shape == sb.isect_offsets.shape:
        result["isect_offsets_exact_eq"] = sa.isect_offsets.equal(sb.isect_offsets)
    else:
        result["isect_offsets_exact_eq"] = False

    # Overall intersection exact equality
    result["intersection_exact_eq"] = (
        result["flatten_ids_exact_eq"] and result["isect_offsets_exact_eq"]
    )

    # radii exact equality (global, all N)
    result["radii_exact_eq"] = radii_a.equal(radii_b)

    return result


def count_consecutive_valid(runs):
    """Given a list of bools, compute consecutive True run lengths."""
    lengths = []
    current = 0
    for v in runs:
        if v:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def main():
    print("=" * 72)
    print("C38: Training-Time Projection/Intersection Reuse Validity Study")
    print("=" * 72)

    device = "cuda"
    torch.manual_seed(42)

    # Config (inline to avoid import issues)
    TILE_SIZE = 16
    PACKED = True
    SH_DEGREE = 3
    EPS2D = 0.1
    RADIUS_CLIP = 0.0
    SPATIAL_LR_SCALE = 46.64

    repo_root = Path(__file__).resolve().parent.parent.parent
    print(f"\nRepo root: {repo_root}")

    # Load dataset
    print("\n  [Loading dataset...]")
    dataset = GTDataset(
        scene="room",
        repo_root=repo_root,
        resolution="1080p",
        device=device,
    )
    print(f"  {len(dataset)} cameras loaded")

    cam0 = dataset.get_camera(0)
    gt0 = dataset.get_gt_image(0)
    print(f"  Camera 0: {cam0.image_width}x{cam0.image_height}")

    # Load model from checkpoint (iter 5000)
    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    print(f"\n  Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=device)
    print(f"  Model: {model.xyz.shape[0]:,} Gaussians, SH degree: {model.sh_degree}")
    print(f"  spatial_lr_scale: {SPATIAL_LR_SCALE}")

    # Setup optimizer (fresh — no optimizer state from checkpoint)
    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * SPATIAL_LR_SCALE,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3,
         "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    # ─── Phase 1: Fixed camera, 500 iterations ───
    N_ITERS = 500
    WARMUP = 50  # let optimizer stabilize

    print(f"\n{'='*60}")
    print(f"Phase 1: {WARMUP} warmup + {N_ITERS} recorded iterations (fixed camera 0)")
    print(f"{'='*60}")

    # Warmup
    print(f"\n  Warmup ({WARMUP} iters)...")
    for i in range(WARMUP):
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=cam0.image_width, height=cam0.image_height,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        rendered_img = rendered[0].clamp(0, 1)
        loss = combined_loss(rendered_img, gt0, lambda_dssim=0.2)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if (i + 1) % 25 == 0:
            print(f"    warmup {i+1}/{WARMUP}  loss={loss.item():.4f}")

    # Recorded iterations
    print(f"\n  Recording {N_ITERS} iterations...")
    states = []
    deltas = []
    timings = []

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for iter_idx in range(1, N_ITERS + 1):
        # Save params BEFORE update (for delta computation)
        old_xyz = model.xyz.detach().clone()
        old_rotations = model.rotations.detach().clone()
        old_scales = model.scales.detach().clone()
        old_opacity = model.opacity.detach().clone()
        old_shs = model.shs.detach().clone()

        data = model.forward()

        ev_start.record()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=cam0.image_width, height=cam0.image_height,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        ev_end.record()
        torch.cuda.synchronize()
        fwd_ms = ev_start.elapsed_time(ev_end)

        rendered_img = rendered[0].clamp(0, 1)

        # Capture state (before gradient update)
        snap = capture_state(model, cam0, type("C", (), {"tile_size": TILE_SIZE, "packed": PACKED, "radius_clip": RADIUS_CLIP, "eps2d": EPS2D})(), model.sh_degree, iter_idx)
        states.append(snap)
        timings.append(fwd_ms)

        # Loss + backward + step
        loss = combined_loss(rendered_img, gt0, lambda_dssim=0.2)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute parameter deltas
        with torch.no_grad():
            d_xyz = (model.xyz - old_xyz).norm(dim=1).mean().item()
            d_rot = (model.rotations - old_rotations).norm(dim=1).mean().item()
            d_scales = (model.scales - old_scales).norm(dim=1).mean().item()
            d_opacity = (model.opacity - old_opacity).abs().mean().item()
            d_shs = (model.shs - old_shs).norm(dim=1).mean().item()
            # Total parameter delta norm
            total_delta = math.sqrt(
                (model.xyz - old_xyz).pow(2).sum().item() +
                (model.rotations - old_rotations).pow(2).sum().item() +
                (model.scales - old_scales).pow(2).sum().item() +
                (model.opacity - old_opacity).pow(2).sum().item() +
                (model.shs - old_shs).pow(2).sum().item()
            )
        deltas.append({
            "iter": iter_idx,
            "d_xyz_mean": d_xyz,
            "d_rot_mean": d_rot,
            "d_scales_mean": d_scales,
            "d_opacity_mean": d_opacity,
            "d_shs_mean": d_shs,
            "total_delta_norm": total_delta,
        })

        if iter_idx % 50 == 0:
            print(f"  iter={iter_idx:4d}  N={snap.n_gaussians:,}  "
                  f"n_vis={snap.n_visible:,}  n_isect={snap.n_intersections:,}  "
                  f"fwd={fwd_ms:.2f}ms  |dxyz|={d_xyz:.3e}")

    # ─── Analysis: K-Step Exact Validity ───
    print(f"\n{'='*60}")
    print("K-Step Exact Validity Analysis")
    print(f"{'='*60}")

    K_VALUES = [1, 2, 4, 8, 16]

    # For each K, check intersection exact equality
    print("\n--- Intersection exact equality (flatten_ids + isect_offsets) ---")
    for K in K_VALUES:
        valid_count = 0
        total_pairs = 0
        for i in range(len(states) - K):
            sa = states[i]
            sb = states[i + K]
            if sa.n_gaussians != sb.n_gaussians:
                continue
            result = compare_states(sa, sb)
            total_pairs += 1
            if result["intersection_exact_eq"]:
                valid_count += 1
        frac = valid_count / max(total_pairs, 1) * 100
        print(f"  K={K:2d}: {valid_count:4d}/{total_pairs:4d} valid ({frac:5.1f}%)")

    # Radii exact equality
    print("\n--- Radii exact equality (visibility) ---")
    for K in K_VALUES:
        valid_count = 0
        total_pairs = 0
        for i in range(len(states) - K):
            sa = states[i]
            sb = states[i + K]
            if sa.n_gaussians != sb.n_gaussians:
                continue
            result = compare_states(sa, sb)
            total_pairs += 1
            if result["radii_exact_eq"]:
                valid_count += 1
        frac = valid_count / max(total_pairs, 1) * 100
        print(f"  K={K:2d}: {valid_count:4d}/{total_pairs:4d} valid ({frac:5.1f}%)")

    # Means2d delta statistics
    print("\n--- Means2d delta (visible-in-both GS, remapped to global) ---")
    for K in K_VALUES:
        max_deltas = []
        mean_deltas = []
        for i in range(len(states) - K):
            sa = states[i]
            sb = states[i + K]
            if sa.n_gaussians != sb.n_gaussians:
                continue
            result = compare_states(sa, sb)
            if result["n_vis_both"] > 0:
                max_deltas.append(result["means2d_max_delta"])
                mean_deltas.append(result["means2d_mean_delta"])
        if max_deltas:
            arr_max = np.array(max_deltas)
            arr_mean = np.array(mean_deltas)
            print(f"  K={K:2d}: max_delta p50={np.percentile(arr_max,50):.3e}  "
                  f"p90={np.percentile(arr_max,90):.3e}  "
                  f"max={arr_max.max():.3e}  |  "
                  f"mean_delta p50={np.percentile(arr_mean,50):.3e}")

    # ─── Consecutive K=1 validity runs ───
    print(f"\n{'='*60}")
    print("Consecutive K=1 Validity Runs (intersection exact equality)")
    print(f"{'='*60}")

    pair_valid = []
    pair_results = []
    for i in range(len(states) - 1):
        sa = states[i]
        sb = states[i + 1]
        if sa.n_gaussians != sb.n_gaussians:
            pair_valid.append(False)
            pair_results.append(None)
            continue
        result = compare_states(sa, sb)
        pair_valid.append(result["intersection_exact_eq"])
        pair_results.append(result)

    run_lengths = count_consecutive_valid(pair_valid)
    print(f"  Total pairwise comparisons: {len(pair_valid)}")
    print(f"  Total valid (K>=1): {sum(pair_valid)} ({sum(pair_valid)/len(pair_valid)*100:.1f}%)")
    if run_lengths:
        lengths = np.array(run_lengths)
        print(f"  P50: {np.percentile(lengths, 50):.0f}")
        print(f"  P90: {np.percentile(lengths, 90):.0f}")
        print(f"  P99: {np.percentile(lengths, 99):.0f}")
        print(f"  Max: {lengths.max()}")
        print(f"  Mean run: {lengths.mean():.1f}")
        print(f"  K>=2 runs: {sum(1 for l in lengths if l >= 2)}/{len(lengths)}")
        print(f"  K>=4 runs: {sum(1 for l in lengths if l >= 4)}/{len(lengths)}")
    else:
        print(f"  No consecutive valid runs found (K always = 0)")

    # ─── Per-pair detail (first 20 pairs) ───
    print(f"\n{'='*60}")
    print("Per-pair detail (first 20 pairs)")
    print(f"{'='*60}")
    print(f"  {'iter':>4s}  {'n_isect':>8s}  {'Δn_isect':>8s}  {'vis_chg':>7s}  "
          f"{'m2d_max':>10s}  {'depth_max':>10s}  {'flat_eq':>7s}  {'off_eq':>7s}  "
          f"{'isect_eq':>7s}  {'|dxyz|':>10s}")
    for i in range(min(20, len(pair_results))):
        r = pair_results[i]
        d = deltas[i]
        if r is None:
            print(f"  {i+1:4d}  --- shape changed ---")
            continue
        print(f"  {i+1:4d}  {r['n_isects_a']:8d}  {r['n_isects_changed']:+8d}  "
              f"{r['n_visibility_changed']:7d}  "
              f"{r['means2d_max_delta']:10.3e}  {r['depths_max_delta']:10.3e}  "
              f"{'Y' if r['flatten_ids_exact_eq'] else 'N':>7s}  "
              f"{'Y' if r['isect_offsets_exact_eq'] else 'N':>7s}  "
              f"{'Y' if r['intersection_exact_eq'] else 'N':>7s}  "
              f"{d['d_xyz_mean']:10.3e}")

    # ─── Parameter delta vs invalidation ───
    print(f"\n{'='*60}")
    print("Parameter Delta vs Intersection Invalidation")
    print(f"{'='*60}")

    delta_arr = np.array([d["d_xyz_mean"] for d in deltas[:-1]])
    total_delta_arr = np.array([d["total_delta_norm"] for d in deltas[:-1]])
    invalid = ~np.array(pair_valid)

    if invalid.sum() > 0 and (~invalid).sum() > 0:
        print(f"\n  When intersection STABLE (rare):")
        print(f"    |dxyz| mean: {delta_arr[~invalid].mean():.3e}  max: {delta_arr[~invalid].max():.3e}")
        print(f"    |Δθ|  mean: {total_delta_arr[~invalid].mean():.3e}  max: {total_delta_arr[~invalid].max():.3e}")
        print(f"  When intersection CHANGED:")
        print(f"    |dxyz| mean: {delta_arr[invalid].mean():.3e}  min: {delta_arr[invalid].min():.3e}")
        print(f"    |Δθ|  mean: {total_delta_arr[invalid].mean():.3e}  min: {total_delta_arr[invalid].min():.3e}")
    elif invalid.all():
        print(f"\n  ALL pairs invalidated (K=0 always)")
        print(f"  Min |dxyz| when invalid: {delta_arr.min():.3e}")
        print(f"  Min |Δθ|  when invalid: {total_delta_arr.min():.3e}")

    # ─── Visibility change stats ───
    print(f"\n{'='*60}")
    print("Visibility Change Statistics (K=1)")
    print(f"{'='*60}")
    vis_changes = [r["n_visibility_changed"] for r in pair_results if r is not None]
    if vis_changes:
        arr = np.array(vis_changes)
        print(f"  Visibility changes per iter: mean={arr.mean():.1f}  median={np.median(arr):.0f}  "
              f"p90={np.percentile(arr,90):.0f}  max={arr.max()}")
        print(f"  Became visible:   mean={np.mean([r['n_became_visible'] for r in pair_results if r]):.1f}")
        print(f"  Became invisible: mean={np.mean([r['n_became_invisible'] for r in pair_results if r]):.1f}")

    # ─── n_isects change stats ───
    print(f"\n{'='*60}")
    print("Intersection Count Change Statistics (K=1)")
    print(f"{'='*60}")
    isect_changes = [abs(r["n_isects_changed"]) for r in pair_results if r is not None]
    if isect_changes:
        arr = np.array(isect_changes)
        print(f"  |Δn_isect| per iter: mean={arr.mean():.1f}  median={np.median(arr):.0f}  "
              f"p90={np.percentile(arr,90):.0f}  max={arr.max()}")
        print(f"  As % of n_isect: {arr.mean()/np.mean([r['n_isects_a'] for r in pair_results if r])*100:.3f}%")

    # ─── Timing ───
    print(f"\n{'='*60}")
    print("Reference Recomputation Cost")
    print(f"{'='*60}")
    timings_arr = np.array(timings)
    print(f"  Forward (project+intersect+rasterize): "
          f"mean={timings_arr.mean():.2f}ms  median={np.median(timings_arr):.2f}ms  "
          f"p90={np.percentile(timings_arr,90):.2f}ms")

    # ─── Phase 2: Camera A/B recurrence ───
    print(f"\n{'='*60}")
    print("Phase 2: Camera A/B Recurrence Test")
    print(f"{'='*60}")

    # Reload model from checkpoint for fair comparison
    model2 = GaussianModel.from_checkpoint_state(
        torch.load(ckpt_path, map_location=device, weights_only=True)["model_state"],
        device=device,
    )
    print(f"  Model2: {model2.xyz.shape[0]:,} Gaussians, SH degree: {model2.sh_degree}")

    optimizer2 = torch.optim.Adam([
        {"params": [model2.xyz], "lr": 1.6e-4 * SPATIAL_LR_SCALE,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model2.rotations], "lr": 1e-3,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model2.scales], "lr": 5e-3,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model2.opacity], "lr": 5e-2,
         "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model2.shs], "lr": 2.5e-3,
         "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    cam1 = dataset.get_camera(1)
    print(f"  Camera 0: {cam0.image_width}x{cam0.image_height}")
    print(f"  Camera 1: {cam1.image_width}x{cam1.image_height}")

    # Schedule: 5×A, 5×B, 5×A, 5×B, ... (100 iters total)
    schedule = []
    for block in range(10):
        cam_id = block % 2  # 0,1,0,1,...
        schedule.extend([cam_id] * 5)

    # Warmup
    for w in range(20):
        cid = schedule[w]
        cam = cam0 if cid == 0 else cam1
        gt = dataset.get_gt_image(cid)
        data = model2.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model2.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        loss = combined_loss(rendered[0].clamp(0, 1), gt, lambda_dssim=0.2)["loss"]
        optimizer2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        optimizer2.step()

    # Record
    states_ab = []
    for iter_idx, cid in enumerate(schedule):
        cam = cam0 if cid == 0 else cam1
        gt = dataset.get_gt_image(cid)
        data = model2.forward()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model2.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        # Handle radii shape: gsplat 1.5.3 may return [nnz, 2] int32 in packed mode
        radii_data_ab = info["radii"].detach().cpu().clone()
        if radii_data_ab.dim() > 1:
            radii_data_ab = radii_data_ab.max(dim=-1).values
        radii_data_ab = radii_data_ab.float()  # ensure float dtype

        # Handle depths shape
        depths_data_ab = info["depths"].detach().cpu().clone()
        if depths_data_ab.dim() > 1:
            depths_data_ab = depths_data_ab.squeeze(-1)
        depths_data_ab = depths_data_ab.float()

        # Handle conics shape
        conics_data_ab = info["conics"].detach().cpu().clone()
        if conics_data_ab.dim() > 1 and conics_data_ab.shape[-1] > 3:
            conics_data_ab = conics_data_ab[..., :3]
        conics_data_ab = conics_data_ab.float()

        snap = StateSnapshot(
            iter=iter_idx,
            means2d_packed=info["means2d"].detach().cpu().clone(),
            conics_packed=conics_data_ab,
            depths_packed=depths_data_ab,
            radii_packed=radii_data_ab,
            gaussian_ids=info["gaussian_ids"].detach().cpu().clone(),
            flatten_ids=info["flatten_ids"].detach().cpu().clone(),
            isect_offsets=info["isect_offsets"].detach().cpu().clone(),
            n_gaussians=data["xyz"].shape[0],
            n_visible=info["means2d"].shape[0],
            n_intersections=info["flatten_ids"].shape[0],
            tile_width=info["tile_width"],
            tile_height=info["tile_height"],
        )
        states_ab.append((snap, cid))

        loss = combined_loss(rendered[0].clamp(0, 1), gt, lambda_dssim=0.2)["loss"]
        optimizer2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        optimizer2.step()

    # Analyze
    print(f"\n  Camera A/B recurrence analysis ({len(states_ab)} iters):")
    same_cam_valid = 0
    same_cam_total = 0
    cross_cam_valid = 0
    cross_cam_total = 0

    for i in range(len(states_ab) - 1):
        sa, ca = states_ab[i]
        sb, cb = states_ab[i + 1]
        if sa.n_gaussians != sb.n_gaussians:
            continue
        result = compare_states(sa, sb)
        is_valid = result["intersection_exact_eq"]

        if ca == cb:
            same_cam_valid += int(is_valid)
            same_cam_total += 1
        else:
            cross_cam_valid += int(is_valid)
            cross_cam_total += 1

    print(f"  Same-camera adjacency:  {same_cam_valid}/{same_cam_total} "
          f"({same_cam_valid/max(same_cam_total,1)*100:.1f}%) intersection exact")
    print(f"  Cross-camera adjacency: {cross_cam_valid}/{cross_cam_total} "
          f"({cross_cam_valid/max(cross_cam_total,1)*100:.1f}%) intersection exact")

    # ─── Summary ───
    print(f"\n{'='*72}")
    print("C38 VALIDITY STUDY SUMMARY")
    print(f"{'='*72}")

    max_k = max(run_lengths) if run_lengths else 0
    print(f"\n  K distribution (consecutive intersection exact equality):")
    print(f"    Total valid pairs (K>=1): {sum(pair_valid)}/{len(pair_valid)} "
          f"({sum(pair_valid)/len(pair_valid)*100:.1f}%)")
    if run_lengths:
        print(f"    Max K: {max_k}")
        print(f"    P50: {np.percentile(lengths, 50):.0f}")
        print(f"    P90: {np.percentile(lengths, 90):.0f}")
        print(f"    K>=2 runs: {sum(1 for l in lengths if l >= 2)}")
        print(f"    K>=4 runs: {sum(1 for l in lengths if l >= 4)}")
    else:
        print(f"    No valid pairs at all (K=0 always)")

    print(f"\n  Means2d max delta (K=1, remapped to global): "
          f"median={np.median([r['means2d_max_delta'] for r in pair_results if r]):.3e}")

    print(f"\n  Visibility changes per iter: "
          f"median={np.median(vis_changes):.0f}  "
          f"({np.median(vis_changes)/np.median([r['n_vis_both'] for r in pair_results if r])*100:.3f}% of visible)")

    print(f"\n  Intersection count change per iter: "
          f"median |Δn_isect|={np.median(isect_changes):.0f}  "
          f"({np.median(isect_changes)/np.median([r['n_isects_a'] for r in pair_results if r])*100:.3f}% of n_isect)")

    print(f"\n  Forward cost: mean={timings_arr.mean():.2f}ms")

    # ─── Final verdict ───
    print(f"\n{'='*72}")
    print("FINAL VERDICT")
    print(f"{'='*72}")
    if max_k >= 4:
        print("  K >= 4 observed → KEEP FOR PROTOTYPE")
    elif max_k >= 2:
        print("  K >= 2 observed → MAYBE")
    elif sum(pair_valid) > 0:
        print("  K=1 occasionally observed → WEAK MAYBE")
    else:
        print("  K=0 always (no exact validity) → DROP")
    print()

    # ─── Save data ───
    output = {
        "pair_valid": [bool(v) for v in pair_valid],
        "consecutive_runs": [int(l) for l in run_lengths],
        "max_k": int(max_k),
        "deltas": deltas,
        "timings": timings,
        "n_gaussians": int(states[0].n_gaussians),
        "n_iters": N_ITERS,
        "warmup": WARMUP,
        "pair_results_summary": [
            {
                "iter": i + 1,
                "n_isects_a": r["n_isects_a"] if r else 0,
                "n_isects_changed": r["n_isects_changed"] if r else 0,
                "n_visibility_changed": r["n_visibility_changed"] if r else 0,
                "means2d_max_delta": r["means2d_max_delta"] if r else 0,
                "depths_max_delta": r["depths_max_delta"] if r else 0,
                "conics_max_delta": r["conics_max_delta"] if r else 0,
                "flatten_ids_exact_eq": bool(r["flatten_ids_exact_eq"]) if r else False,
                "isect_offsets_exact_eq": bool(r["isect_offsets_exact_eq"]) if r else False,
                "intersection_exact_eq": bool(r["intersection_exact_eq"]) if r else False,
            }
            for i, r in enumerate(pair_results)
        ],
    }

    save_path = Path("results/phase-c31/c38_validity_data.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
