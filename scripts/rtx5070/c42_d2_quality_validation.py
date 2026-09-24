#!/usr/bin/env python3
"""
C42 Phase D-2: End-to-End Quality Validation.

Compares Baseline (SSIM scale=1.0) vs Candidate (SSIM scale=0.5).

Trains from iter-5000 checkpoint for 1000, 3000, 5000, 10000 additional
iterations, measuring PSNR, SSIM, loss, Gaussian count, training time,
and rotation parameter statistics at each interval.

Uses full training pipeline: round-robin cameras, densification, pruning,
SH scheduling, gradient clipping — matching train_3dgs.py.
"""
import json, math, sys, time, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset

DEVICE = "cuda"
TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2
SPATIAL_LR_SCALE = 46.64

# Training config (matching train_3dgs.py defaults)
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

# Evaluation checkpoints (additional iterations from resume point)
EVAL_ITERS = [1000, 3000, 5000, 10000]
EVAL_CAMERAS = list(range(0, 311, 25))  # 13 cameras for eval
WARMUP_ITERS = 50


def d_ssim_downsampled(pred, target, scale, window_size=11, sigma=1.5, data_range=1.0):
    """D-SSIM with optional downsampling."""
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


def combined_loss_ds(pred, target, scale, lambda_dssim=LAMBDA_DSSIM):
    l1 = F.l1_loss(pred, target)
    dsim = d_ssim_downsampled(pred, target, scale=scale)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dsim, l1, dsim


def make_optimizer(model):
    return torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * SPATIAL_LR_SCALE, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    ])


def evaluate_quality(model, dataset, cam_indices):
    """Evaluate PSNR, SSIM, L1, D-SSIM across cameras."""
    psnrs, ssims, l1s, dsims = [], [], [], []
    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        with torch.no_grad():
            data = model.forward()
            rendered, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
                radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
            )
            pred = rendered[0].clamp(0, 1)

            mse = float(((pred - gt) ** 2).mean())
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
            psnrs.append(psnr)

            l1 = float(F.l1_loss(pred, gt))
            l1s.append(l1)

            dsim = float(d_ssim_downsampled(pred, gt, scale=1.0))  # eval SSIM at full res
            dsims.append(dsim)
            ssims.append(1.0 - dsim)

    return {
        "psnr": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "l1": float(np.mean(l1s)),
        "d_ssim": float(np.mean(dsims)),
        "n_cameras": len(cam_indices),
    }


def rotation_stats(model):
    """Compute rotation parameter statistics."""
    rot = model.rotations.detach()
    norms = rot.norm(dim=-1)
    return {
        "rot_mean_norm": float(norms.mean()),
        "rot_std_norm": float(norms.std()),
        "rot_mean_abs": float(rot.abs().mean()),
        "rot_std": float(rot.std()),
        "rot_grad_norm": float(rot.grad.norm().item()) if rot.grad is not None else 0.0,
        "rot_entropy": float(-(F.softmax(rot, dim=-1) * F.log_softmax(rot, dim=-1)).sum(dim=-1).mean()),
    }


def train_variant(model, dataset, scale, n_iters, resume_iter, eval_cameras):
    """Train one variant and evaluate at checkpoints."""
    optimizer = make_optimizer(model)
    num_cameras = len(dataset)

    results = []

    # Evaluate at start (iter 0 = resume point)
    print(f"    [eval@0] ", end="", flush=True)
    q = evaluate_quality(model, dataset, eval_cameras)
    q["iteration"] = resume_iter
    q["additional_iters"] = 0
    q["n_gaussians"] = model.xyz.shape[0]
    q["rotation_stats"] = rotation_stats(model)
    results.append(q)
    print(f"PSNR={q['psnr']:.2f}  SSIM={q['ssim']:.4f}  GS={q['n_gaussians']:,}")

    # Warmup
    for i in range(WARMUP_ITERS):
        cam_idx = (resume_iter + i) % num_cameras
        cam = dataset.get_camera(cam_idx)
        gt = dataset.get_gt_image(cam_idx)
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        pred = rendered[0].clamp(0, 1)
        loss, l1_val, dsim_val = combined_loss_ds(pred, gt, scale=scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    torch.cuda.synchronize()

    # Timed training
    iter_times = []
    next_eval = 0
    eval_idx = 0

    for i in range(1, n_iters + 1):
        iteration = resume_iter + WARMUP_ITERS + i
        cam_idx = iteration % num_cameras
        cam = dataset.get_camera(cam_idx)
        gt = dataset.get_gt_image(cam_idx)

        # SH degree scheduling (won't trigger at iter 5000+, max degree already 3)
        new_degree = min(MAX_SH_DEGREE, iteration // SH_DEGREE_INTERVAL)
        if new_degree != model.sh_degree and new_degree <= MAX_SH_DEGREE:
            model.set_sh_degree(new_degree)
            optimizer = make_optimizer(model)  # discard old state (matches train_3dgs.py)

        t_start = time.perf_counter()

        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        pred = rendered[0].clamp(0, 1)

        loss, l1_val, dsim_val = combined_loss_ds(pred, gt, scale=scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Positional gradient accumulation for densification
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

        if denf_count["cloned"] + denf_count["split"] + prune_count > 0:
            optimizer = make_optimizer(model)  # discard old state (matches train_3dgs.py)

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t_start) * 1000
        iter_times.append(iter_ms)

        # Evaluate at checkpoint
        if eval_idx < len(EVAL_ITERS) and i >= EVAL_ITERS[eval_idx]:
            print(f"    [eval@{EVAL_ITERS[eval_idx]}] ", end="", flush=True)
            q = evaluate_quality(model, dataset, eval_cameras)
            q["iteration"] = iteration
            q["additional_iters"] = EVAL_ITERS[eval_idx]
            q["n_gaussians"] = model.xyz.shape[0]
            q["rotation_stats"] = rotation_stats(model)
            q["mean_iter_ms"] = float(np.mean(iter_times[-EVAL_ITERS[eval_idx]:]))
            q["total_train_time_s"] = float(np.sum(iter_times) / 1000)
            results.append(q)
            print(f"PSNR={q['psnr']:.2f}  SSIM={q['ssim']:.4f}  GS={q['n_gaussians']:,}  "
                  f"iter={q['mean_iter_ms']:.1f}ms")
            eval_idx += 1

    # Final timing summary
    final_timing = {
        "mean_iter_ms": float(np.mean(iter_times)),
        "std_iter_ms": float(np.std(iter_times)),
        "total_time_s": float(np.sum(iter_times) / 1000),
        "n_iters": n_iters,
    }
    return results, final_timing


def main():
    print("=" * 72)
    print("C42 Phase D-2: End-to-End Quality Validation")
    print("Baseline (scale=1.0) vs Candidate (scale=0.5)")
    print("=" * 72)

    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parent.parent.parent

    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    num_cameras = len(dataset)
    print(f"  Dataset: {num_cameras} cameras")

    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    resume_iter = ckpt.get("iteration", 5000)

    eval_cameras = EVAL_CAMERAS
    n_train_iters = max(EVAL_ITERS)
    print(f"  Checkpoint: iter {resume_iter}, {ckpt['model_state']['xyz'].shape[0]:,} Gaussians")
    print(f"  Training: {n_train_iters} additional iterations")
    print(f"  Eval cameras: {len(eval_cameras)}")

    # ── Run Baseline ──
    print(f"\n{'='*60}")
    print("BASELINE: SSIM scale=1.0")
    print(f"{'='*60}")

    model_base = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=DEVICE)
    base_results, base_timing = train_variant(
        model_base, dataset, scale=1.0, n_iters=n_train_iters,
        resume_iter=resume_iter, eval_cameras=eval_cameras
    )
    del model_base
    torch.cuda.empty_cache()

    # ── Run Candidate ──
    print(f"\n{'='*60}")
    print("CANDIDATE: SSIM scale=0.5")
    print(f"{'='*60}")

    model_cand = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=DEVICE)
    cand_results, cand_timing = train_variant(
        model_cand, dataset, scale=0.5, n_iters=n_train_iters,
        resume_iter=resume_iter, eval_cameras=eval_cameras
    )
    del model_cand
    torch.cuda.empty_cache()

    # ── Comparison ──
    print(f"\n{'='*72}")
    print("COMPARISON: Baseline vs Candidate")
    print(f"{'='*72}")

    print(f"\n  {'Add iters':>9s}  {'PSNR_b':>7s}  {'PSNR_c':>7s}  {'dPSNR':>7s}  "
          f"{'SSIM_b':>7s}  {'SSIM_c':>7s}  {'dSSIM':>7s}  "
          f"{'GS_b':>8s}  {'GS_c':>8s}  "
          f"{'iter_b':>7s}  {'iter_c':>7s}  {'speedup':>7s}")
    print("  " + "-" * 110)

    comparison = []
    for br, cr in zip(base_results, cand_results):
        d_psnr = cr["psnr"] - br["psnr"]
        d_ssim = cr["ssim"] - br["ssim"]
        speedup = (1 - cr.get("mean_iter_ms", 0) / br.get("mean_iter_ms", 1)) * 100 if br.get("mean_iter_ms", 0) > 0 else 0

        print(f"  {br['additional_iters']:>9d}  {br['psnr']:>7.2f}  {cr['psnr']:>7.2f}  {d_psnr:>+7.2f}  "
              f"{br['ssim']:>7.4f}  {cr['ssim']:>7.4f}  {d_ssim:>+7.4f}  "
              f"{br['n_gaussians']:>8,}  {cr['n_gaussians']:>8,}  "
              f"{br.get('mean_iter_ms', 0):>7.1f}  {cr.get('mean_iter_ms', 0):>7.1f}  {speedup:>+6.1f}%")

        comparison.append({
            "additional_iters": br["additional_iters"],
            "baseline": br,
            "candidate": cr,
            "d_psnr": d_psnr,
            "d_ssim": d_ssim,
            "speedup_pct": speedup,
        })

    # ── Rotation stats comparison ──
    print(f"\n  Rotation parameter statistics:")
    print(f"  {'Iters':>6s}  {'rot_norm_b':>10s}  {'rot_norm_c':>10s}  "
          f"{'rot_std_b':>10s}  {'rot_std_c':>10s}  "
          f"{'grad_b':>10s}  {'grad_c':>10s}")
    print("  " + "-" * 75)
    for br, cr in zip(base_results, cand_results):
        rb, rc = br["rotation_stats"], cr["rotation_stats"]
        print(f"  {br['additional_iters']:>6d}  {rb['rot_mean_norm']:>10.4f}  {rc['rot_mean_norm']:>10.4f}  "
              f"{rb['rot_std']:>10.4f}  {rc['rot_std']:>10.4f}  "
              f"{rb['rot_grad_norm']:>10.6f}  {rc['rot_grad_norm']:>10.6f}")

    # ── Decision ──
    print(f"\n{'='*72}")
    print("DECISION")
    print(f"{'='*72}")

    # Use the final checkpoint (max iters) for decision
    final = comparison[-1]
    print(f"\n  Final checkpoint ({final['additional_iters']} additional iters):")
    print(f"    PSNR drop:     {final['d_psnr']:+.3f} dB  (gate: < -0.1 dB)")
    print(f"    SSIM drop:     {final['d_ssim']:+.5f}  (gate: < -0.005)")
    print(f"    Speedup:       {final['speedup_pct']:+.1f}%  (gate: > 25%)")

    psnr_pass = final["d_psnr"] > -0.1
    ssim_pass = final["d_ssim"] > -0.005
    speedup_pass = final["speedup_pct"] > 25

    if psnr_pass and ssim_pass and speedup_pass:
        decision = "KEEP"
    elif speedup_pass and (psnr_pass or ssim_pass):
        decision = "MAYBE (one quality gate failed)"
    elif speedup_pass:
        decision = "MAYBE (quality gates failed but speedup strong)"
    else:
        decision = "DROP"

    print(f"\n  Decision: {decision}")

    # ── Save ──
    output = {
        "experiment": "C42 Phase D-2: End-to-End Quality Validation",
        "config": {
            "scene": "room",
            "resume_iter": resume_iter,
            "eval_iters": EVAL_ITERS,
            "eval_cameras": eval_cameras,
            "n_train_iters": n_train_iters,
            "lambda_dssim": LAMBDA_DSSIM,
        },
        "baseline_scale_1.0": {
            "results": base_results,
            "timing": base_timing,
        },
        "candidate_scale_0.5": {
            "results": cand_results,
            "timing": cand_timing,
        },
        "comparison": comparison,
        "decision": decision,
    }

    save_path = Path("results/phase-c42/c42_d2_quality_validation.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
