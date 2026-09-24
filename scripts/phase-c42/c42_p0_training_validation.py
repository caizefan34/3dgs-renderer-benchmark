#!/usr/bin/env python3
"""
C42 P0 Training Validation: Baseline vs Downsampled SSIM (scale=0.5).

Runs two identical training runs on A100-PCIE-40GB:
  A: 0.8*L1(full) + 0.2*D-SSIM(full)     — baseline
  B: 0.8*L1(full) + 0.2*D-SSIM(scale=0.5) — candidate

Same: scene, seed, SfM init, optimizer, densification, pruning, SH schedule, camera sampling.
Different: only the D-SSIM resolution.

Records: wall-clock time, PSNR (every 500), SSIM (every 1000), Gaussian count, L1, D-SSIM.
"""
import json, math, sys, time
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
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"
TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2
SEED = 42
NUM_ITERS = 10000
PSNR_INTERVAL = 500
SSIM_INTERVAL = 1000
LOSS_INTERVAL = 100
GCOUNT_INTERVAL = 500

# Training config (matching train_3dgs.py defaults exactly)
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

# Evaluation cameras (every 25th of 311)
EVAL_CAMERAS = list(range(0, 311, 25))  # 13 cameras


def d_ssim_loss(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    """Standard D-SSIM loss at full resolution."""
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


def d_ssim_downsampled(pred, target, scale=0.5, window_size=11, sigma=1.5, data_range=1.0):
    """D-SSIM loss with downsampling before computation."""
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
    """Combined loss: 0.8*L1(full) + 0.2*D-SSIM(scale)."""
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


def evaluate(model, dataset, cam_indices):
    """Evaluate PSNR and SSIM across cameras."""
    psnrs, ssims = [], []
    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        with torch.no_grad():
            pred = render(model, cam)
            mse = float(((pred - gt) ** 2).mean())
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
            psnrs.append(psnr)
            # Full-resolution SSIM (always at scale=1.0 for evaluation)
            ssim_val = 1.0 - float(d_ssim_loss(pred, gt))
            ssims.append(ssim_val)
    return float(np.mean(psnrs)), float(np.mean(ssims))


def train_variant(dataset, sfm_data, scale, variant_name):
    """Run one training variant."""
    print(f"\n{'='*60}")
    print(f"  VARIANT {variant_name}: SSIM scale={scale}")
    print(f"{'='*60}")

    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Initialize model from SfM
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
        "variant": variant_name,
        "scale": scale,
        "seed": SEED,
        "per_iter": [],
        "eval_points": [],
        "final_eval": None,
    }

    # Initial eval
    print(f"  [eval@0] ", end="", flush=True)
    psnr, ssim = evaluate(model, dataset, EVAL_CAMERAS)
    ng = model.xyz.shape[0]
    results["eval_points"].append({"iter": 0, "psnr": psnr, "ssim": ssim, "n_gaussians": ng})
    print(f"PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={ng:,}")

    iter_times = []
    t_start_total = time.perf_counter()

    for iteration in range(1, NUM_ITERS + 1):
        # Camera selection (round-robin, same as train_3dgs.py)
        cam_idx = (iteration - 1) % num_cameras
        cam = dataset.get_camera(cam_idx)
        gt = dataset.get_gt_image(cam_idx)

        # SH degree scheduling
        new_degree = min(MAX_SH_DEGREE, iteration // SH_DEGREE_INTERVAL)
        if new_degree != model.sh_degree and new_degree <= MAX_SH_DEGREE:
            model.set_sh_degree(new_degree)
            optimizer = make_optimizer(model, spatial_lr_scale)

        t0 = time.perf_counter()

        # Forward
        pred = render(model, cam)

        # Loss
        loss, l1_val, dsim_val = compute_loss(pred, gt, scale)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Positional gradient accumulation
        model.accumulate_positional_gradient()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
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

        # Rebuild optimizer if topology changed
        if denf_count["cloned"] + denf_count["split"] + prune_count > 0:
            optimizer = make_optimizer(model, spatial_lr_scale)

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000
        iter_times.append(iter_ms)

        # Logging
        if iteration % LOSS_INTERVAL == 0:
            results["per_iter"].append({
                "iter": iteration,
                "loss": float(loss.item()),
                "l1": float(l1_val.item()),
                "d_ssim": float(dsim_val.item()),
                "iter_ms": iter_ms,
                "n_gaussians": model.xyz.shape[0],
            })

        # PSNR eval
        if iteration % PSNR_INTERVAL == 0:
            psnr, ssim = evaluate(model, dataset, EVAL_CAMERAS)
            ng = model.xyz.shape[0]
            mean_iter_ms = float(np.mean(iter_times[-PSNR_INTERVAL:]))
            results["eval_points"].append({
                "iter": iteration, "psnr": psnr, "ssim": ssim,
                "n_gaussians": ng, "mean_iter_ms": mean_iter_ms,
            })
            print(f"  [eval@{iteration}] PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={ng:,}  "
                  f"iter={mean_iter_ms:.1f}ms")

        # SSIM eval (at SSIM_INTERVAL, but PSNR eval already includes SSIM)
        # If PSNR_INTERVAL != SSIM_INTERVAL, add separate SSIM-only eval
        elif iteration % SSIM_INTERVAL == 0:
            psnr, ssim = evaluate(model, dataset, EVAL_CAMERAS)
            ng = model.xyz.shape[0]
            results["eval_points"].append({
                "iter": iteration, "psnr": psnr, "ssim": ssim,
                "n_gaussians": ng,
            })
            print(f"  [eval@{iteration}] PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={ng:,}")

    total_time = time.perf_counter() - t_start_total

    # Final eval (all 311 cameras)
    print(f"  [final eval, all 311 cams] ", end="", flush=True)
    all_cams = list(range(311))
    psnr, ssim = evaluate(model, dataset, all_cams)
    ng = model.xyz.shape[0]
    results["final_eval"] = {
        "iter": NUM_ITERS, "psnr": psnr, "ssim": ssim,
        "n_gaussians": ng, "n_eval_cameras": 311,
    }
    print(f"PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={ng:,}")

    # Timing summary
    results["timing"] = {
        "total_wall_s": total_time,
        "mean_iter_ms": float(np.mean(iter_times)),
        "std_iter_ms": float(np.std(iter_times)),
        "min_iter_ms": float(np.min(iter_times)),
        "max_iter_ms": float(np.max(iter_times)),
        "n_iters": NUM_ITERS,
    }
    print(f"\n  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Mean iter: {np.mean(iter_times):.2f} ms")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 72)
    print("C42 P0 Training Validation: Baseline vs Downsampled SSIM")
    print("=" * 72)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {gpu_name}")
    print(f"  SMs: {gpu_props.multi_processor_count}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    import gsplat
    print(f"  gsplat: {gsplat.__version__}")
    print(f"  Seed: {SEED}, Iters: {NUM_ITERS}")

    repo_root = Path(__file__).resolve().parent.parent.parent

    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    print(f"  Cameras: {len(dataset)}")

    print("\n  [Loading SfM checkpoint...]")
    sfm_data = load_initial_checkpoint("room", repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    # ── Run Variant A (baseline) ──
    results_a = train_variant(dataset, sfm_data, scale=1.0, variant_name="A_baseline")
    torch.cuda.empty_cache()

    # ── Run Variant B (scale=0.5) ──
    results_b = train_variant(dataset, sfm_data, scale=0.5, variant_name="B_downsampled_0.5")
    torch.cuda.empty_cache()

    # ── Comparison ──
    print(f"\n{'='*72}")
    print("COMPARISON: A (baseline) vs B (scale=0.5)")
    print(f"{'='*72}")

    print(f"\n  {'Iter':>6s}  {'PSNR_A':>7s}  {'PSNR_B':>7s}  {'dPSNR':>7s}  "
          f"{'SSIM_A':>7s}  {'SSIM_B':>7s}  {'dSSIM':>7s}  "
          f"{'GS_A':>8s}  {'GS_B':>8s}  {'dGS%':>6s}")
    print("  " + "-" * 90)

    for ea, eb in zip(results_a["eval_points"], results_b["eval_points"]):
        d_psnr = eb["psnr"] - ea["psnr"]
        d_ssim = eb["ssim"] - ea["ssim"]
        d_gs = (eb["n_gaussians"] - ea["n_gaussians"]) / ea["n_gaussians"] * 100
        print(f"  {ea['iter']:>6d}  {ea['psnr']:>7.2f}  {eb['psnr']:>7.2f}  {d_psnr:>+7.2f}  "
              f"{ea['ssim']:>7.4f}  {eb['ssim']:>7.4f}  {d_ssim:>+7.4f}  "
              f"{ea['n_gaussians']:>8,}  {eb['n_gaussians']:>8,}  {d_gs:>+5.1f}%")

    # Final comparison
    fa, fb = results_a["final_eval"], results_b["final_eval"]
    d_psnr = fb["psnr"] - fa["psnr"]
    d_ssim = fb["ssim"] - fa["ssim"]
    d_gs = (fb["n_gaussians"] - fa["n_gaussians"]) / fa["n_gaussians"] * 100
    speedup = (1 - results_b["timing"]["mean_iter_ms"] / results_a["timing"]["mean_iter_ms"]) * 100

    print(f"\n  FINAL (311 cams, iter {NUM_ITERS}):")
    print(f"    PSNR:  A={fa['psnr']:.2f}  B={fb['psnr']:.2f}  dPSNR={d_psnr:+.2f} dB  (gate: > -0.2)")
    print(f"    SSIM:  A={fa['ssim']:.4f}  B={fb['ssim']:.4f}  dSSIM={d_ssim:+.4f}  (gate: > -0.005)")
    print(f"    GS:    A={fa['n_gaussians']:,}  B={fb['n_gaussians']:,}  dGS={d_gs:+.1f}%  (gate: < 10%)")
    print(f"    Speed: A={results_a['timing']['mean_iter_ms']:.1f}ms  B={results_b['timing']['mean_iter_ms']:.1f}ms  "
          f"speedup={speedup:+.1f}%  (gate: > 40%)")

    # Decision
    psnr_pass = d_psnr > -0.2
    ssim_pass = d_ssim > -0.005
    gs_pass = abs(d_gs) < 10
    speedup_pass = speedup > 40

    print(f"\n  DECISION GATES:")
    print(f"    PSNR drop < 0.2 dB:    {'PASS' if psnr_pass else 'FAIL'} ({d_psnr:+.2f})")
    print(f"    SSIM drop < 0.005:     {'PASS' if ssim_pass else 'FAIL'} ({d_ssim:+.4f})")
    print(f"    GS diff < 10%:         {'PASS' if gs_pass else 'FAIL'} ({d_gs:+.1f}%)")
    print(f"    Speedup > 40%:         {'PASS' if speedup_pass else 'FAIL'} ({speedup:+.1f}%)")

    if psnr_pass and ssim_pass and gs_pass and speedup_pass:
        decision = "PASS — all gates met. Prepare 30K full validation."
    else:
        failed = [g for g, p in [("PSNR", psnr_pass), ("SSIM", ssim_pass),
                                  ("GS", gs_pass), ("Speedup", speedup_pass)] if not p]
        decision = f"FAIL — gates not met: {', '.join(failed)}"

    print(f"\n  DECISION: {decision}")

    # Save
    output = {
        "experiment": "C42 P0 Training Validation",
        "hardware": {"gpu": gpu_name, "sms": gpu_props.multi_processor_count,
                     "pytorch": torch.__version__, "cuda": torch.version.cuda},
        "config": {"seed": SEED, "num_iters": NUM_ITERS, "scene": "room",
                   "lambda_dssim": LAMBDA_DSSIM, "tile_size": TILE_SIZE,
                   "n_sfm_points": sfm_data["xyz"].shape[0],
                   "eval_cameras": EVAL_CAMERAS},
        "variant_A_baseline": results_a,
        "variant_B_downsampled_0.5": results_b,
        "comparison": {
            "final_d_psnr": d_psnr, "final_d_ssim": d_ssim,
            "final_d_gs_pct": d_gs, "speedup_pct": speedup,
            "psnr_pass": psnr_pass, "ssim_pass": ssim_pass,
            "gs_pass": gs_pass, "speedup_pass": speedup_pass,
            "decision": decision,
        },
    }
    save_path = repo_root / "results" / "a100" / "phase-c42" / "c42_p0_training_validation.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
