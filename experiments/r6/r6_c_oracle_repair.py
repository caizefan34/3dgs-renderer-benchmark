#!/usr/bin/env python3
"""
R6-C ORACLE REPAIR — correct traffic model with three bounds.

The original r6_5_fusion_oracle.py had a fundamental error:
  T_C_conservative = T_opt_mean + 0.5 * T_avoidable_traffic_ms
This ADDS traffic time to optimizer time, producing a number LARGER than
T_optimizer — impossible for a "savings" metric.

This script computes the CORRECT fusion opportunity:

Traffic model (without fusion):
  Backward writes: gradient → global memory (param .grad buffers)
  Optimizer reads: param, gradient, exp_avg, exp_avg_sq
  Optimizer writes: param, exp_avg, exp_avg_sq

Fusion eliminates:
  - The gradient write (backward → global memory)
  - The gradient read (optimizer ← global memory)
  The gradient is consumed in-register during the fused backward-optimizer step.

The optimizer STILL must:
  - Read param, exp_avg, exp_avg_sq
  - Write param, exp_avg, exp_avg_sq
  These are NOT eliminated by fusion.

Three bounds:
  LOWER: Only eliminate gradient READ by optimizer (write still needed for autograd)
    T_saved = grad_bytes / bandwidth_achieved
    
  CONSERVATIVE: Eliminate both gradient write + read, at achievable bandwidth
    T_saved = 2 × grad_bytes / bandwidth_achieved
    bandwidth_achieved = 1.0 TB/s (A100, ~65% of peak)
    
  OPTIMISTIC: Eliminate ALL intermediate gradient traffic (including VJP chain
    intermediates) at peak bandwidth
    T_saved = 2 × (intermediate + param) / bandwidth_peak
    bandwidth_peak = 1.55 TB/s

The KEY correction: T_saved is bandwidth-limited traffic time, NOT a fraction
of T_optimizer. T_optimizer includes computation that fusion cannot eliminate.

Usage (on mx):
  CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/r6_c_oracle_repair.py \
    --scene room --ckpt /mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_5000.pt \
    --output /mnt/storage_pool/liaoyuanjun/r6_profiling/r6_c_repair_room_5k.json
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
from trainer import SepSSIM, render_with_meta


def load_model_from_ckpt(ckpt_path, config, repo_root):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {
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
    model.active_sh_degree = ckpt["active_sh_degree"]
    return model


def stats_ms(values):
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}


def compute_repaired_oracle(model, cam, gt_image, ssim_fn, sh_degree, n_warmup=20, n_measure=100):
    """Compute the REPAIRED R6-C fusion oracle with proper traffic model."""
    opt_start = torch.cuda.Event(enable_timing=True)
    opt_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    opt_times, bwd_times, iter_times = [], [], []

    # Warmup
    for _ in range(n_warmup):
        model.optimizer.zero_grad(set_to_none=True)
        image, meta, means2d = render_with_meta(model, cam, sh_degree)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - 0.2) * L1 + 0.2 * dssim
        loss.backward()
        model.update_learning_rate(1)
        model.optimizer.step()
        torch.cuda.synchronize()

    # Measure
    for _ in range(n_measure):
        model.optimizer.zero_grad(set_to_none=True)
        iter_start.record()
        image, meta, means2d = render_with_meta(model, cam, sh_degree)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - 0.2) * L1 + 0.2 * dssim
        bwd_start.record()
        loss.backward()
        bwd_end.record()
        torch.cuda.synchronize()
        bwd_ms = bwd_start.elapsed_time(bwd_end)
        model.update_learning_rate(1)
        opt_start.record()
        model.optimizer.step()
        opt_end.record()
        torch.cuda.synchronize()
        opt_ms = opt_start.elapsed_time(opt_end)
        iter_end.record()
        torch.cuda.synchronize()
        iter_ms = iter_start.elapsed_time(iter_end)
        opt_times.append(opt_ms)
        bwd_times.append(bwd_ms)
        iter_times.append(iter_ms)

    T_opt_mean = float(np.mean(opt_times))
    T_bwd_mean = float(np.mean(bwd_times))
    T_iter_mean = float(np.mean(iter_times))

    # === Traffic model ===
    N = model._xyz.shape[0]
    K_active = (sh_degree + 1) ** 2  # 16 for SH degree 3

    # Parameter sizes (what the optimizer reads/writes)
    param_sizes = {
        "xyz": N * 3 * 4,           # [N, 3] float32
        "shs": N * K_active * 3 * 4, # [N, K, 3]
        "scaling": N * 3 * 4,        # [N, 3]
        "rotation": N * 4 * 4,       # [N, 4]
        "opacity": N * 4,            # [N]
    }
    total_param_bytes = sum(param_sizes.values())

    # Gradient bytes = same as parameter bytes (one .grad per parameter)
    grad_bytes = total_param_bytes

    # Intermediate gradient buffers (within backward VJP chain)
    # These are written by rasterizer backward, read by projection/SH backward
    # Fusion of the ENTIRE backward chain could eliminate these
    rasterizer_grad_bytes = N * 11 * 4   # 5 buffers: v_means2d(2), v_conics(3), v_colors(3), v_opac(1), v_means2d_abs(2) → actually 11 floats
    projection_grad_bytes = N * 10 * 4   # 3 buffers: v_means(3), v_quats(4), v_scales(3)
    sh_grad_bytes = N * K_active * 3 * 4 # 1 buffer: v_coefficients
    intermediate_grad_bytes = rasterizer_grad_bytes + projection_grad_bytes + sh_grad_bytes

    # Adam optimizer traffic (without fusion):
    # Reads: grad(1×) + param(1×) + exp_avg(1×) + exp_avg_sq(1×) = 4 reads
    # Writes: param(1×) + exp_avg(1×) + exp_avg_sq(1×) = 3 writes
    # Total: 7 × param_bytes
    adam_total_traffic = 7 * total_param_bytes

    # Gradient traffic within Adam: 1 read = grad_bytes = total_param_bytes
    grad_fraction_of_adam = grad_bytes / adam_total_traffic  # 1/7 = 14.3%

    # === Bandwidth models ===
    A100_PEAK_BW = 1.55e12   # 1.55 TB/s peak HBM bandwidth
    A100_ACHIEVED_BW = 1.0e12  # ~65% of peak (realistic for scattered access)

    # === Three bounds ===

    # LOWER: Only eliminate gradient READ by optimizer
    # (gradient write still needed for autograd bookkeeping)
    avoidable_bytes_lower = grad_bytes
    T_saved_lower = avoidable_bytes_lower / A100_ACHIEVED_BW * 1000  # ms

    # CONSERVATIVE: Eliminate both gradient write (backward) + read (optimizer)
    # at achievable bandwidth
    avoidable_bytes_cons = 2 * grad_bytes
    T_saved_cons = avoidable_bytes_cons / A100_ACHIEVED_BW * 1000

    # OPTIMISTIC: Eliminate ALL gradient traffic (param .grad + intermediate buffers)
    # at peak bandwidth. This requires fusing the ENTIRE backward chain, not just
    # backward-optimizer boundary.
    avoidable_bytes_opt = 2 * (grad_bytes + intermediate_grad_bytes)
    T_saved_opt = avoidable_bytes_opt / A100_PEAK_BW * 1000

    # === Sanity checks ===
    # T_saved must not exceed the physically removable work
    # The gradient read is 1/7 of Adam traffic → T_saved ≤ T_opt/7 (traffic-bound)
    T_saved_upper_physical = T_opt_mean * grad_fraction_of_adam
    T_saved_lower = min(T_saved_lower, T_saved_upper_physical)
    T_saved_cons = min(T_saved_cons, 2 * T_saved_upper_physical)

    return {
        "N_total": N,
        "K_active": K_active,
        "T_bwd_ms": stats_ms(bwd_times),
        "T_optimizer_ms": stats_ms(opt_times),
        "T_iter_ms": stats_ms(iter_times),
        # Traffic model
        "param_sizes_bytes": param_sizes,
        "total_param_bytes": total_param_bytes,
        "grad_bytes": grad_bytes,
        "intermediate_grad_bytes": intermediate_grad_bytes,
        "adam_total_traffic_bytes": adam_total_traffic,
        "grad_fraction_of_adam": grad_fraction_of_adam,
        # Three bounds
        "avoidable_bytes_lower": avoidable_bytes_lower,
        "avoidable_bytes_conservative": avoidable_bytes_cons,
        "avoidable_bytes_optimistic": avoidable_bytes_opt,
        "T_saved_lower_ms": T_saved_lower,
        "T_saved_conservative_ms": T_saved_cons,
        "T_saved_optimistic_ms": T_saved_opt,
        "T_saved_lower_pct_iter": T_saved_lower / T_iter_mean * 100,
        "T_saved_conservative_pct_iter": T_saved_cons / T_iter_mean * 100,
        "T_saved_optimistic_pct_iter": T_saved_opt / T_iter_mean * 100,
        # Comparison with original (broken) oracle
        "T_saved_upper_physical_ms": T_saved_upper_physical,
        "T_saved_upper_physical_pct_iter": T_saved_upper_physical / T_iter_mean * 100,
        # Bandwidth assumptions
        "A100_peak_bw_TBs": A100_PEAK_BW / 1e12,
        "A100_achieved_bw_TBs": A100_ACHIEVED_BW / 1e12,
        # Original oracle values for comparison
        "original_T_C_conservative_pct_iter": (T_opt_mean + 0.5 * (2 * intermediate_grad_bytes / A100_PEAK_BW * 1000)) / T_iter_mean * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=100)
    args = parser.parse_args()

    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = str(REPO_ROOT)
    config.resolution = args.resolution
    config.seed = args.seed

    print(f"=== R6-C Oracle REPAIR: {args.scene} ===")
    dataset = GTDataset(scene=config.scene, repo_root=config.repo_root,
                        resolution=config.resolution, device="cuda", background="black")
    model = load_model_from_ckpt(args.ckpt, config, config.repo_root)
    print(f"  N={model._xyz.shape[0]}, SH={model.active_sh_degree}")

    cam, gt_image = dataset.get_item(0)
    ssim_fn = SepSSIM(device="cuda")

    result = compute_repaired_oracle(model, cam, gt_image, ssim_fn,
                                     model.active_sh_degree, args.n_warmup, args.n_measure)
    result["scene"] = args.scene
    result["checkpoint"] = args.ckpt

    print(f"\n=== REPAIRED Oracle: {args.scene} ===")
    print(f"  T_bwd: {result['T_bwd_ms']['mean']:.2f}ms")
    print(f"  T_optimizer: {result['T_optimizer_ms']['mean']:.2f}ms")
    print(f"  T_iter: {result['T_iter_ms']['mean']:.2f}ms")
    print(f"  Grad bytes: {result['grad_bytes']/1e6:.1f}MB")
    print(f"  Intermediate grad bytes: {result['intermediate_grad_bytes']/1e6:.1f}MB")
    print(f"  Adam total traffic: {result['adam_total_traffic_bytes']/1e6:.1f}MB")
    print(f"  Grad fraction of Adam: {result['grad_fraction_of_adam']*100:.1f}%")
    print(f"  --- Three bounds (T_saved) ---")
    print(f"  LOWER:      {result['T_saved_lower_ms']:.3f}ms ({result['T_saved_lower_pct_iter']:.2f}% T_iter)")
    print(f"  CONSERVATIVE: {result['T_saved_conservative_ms']:.3f}ms ({result['T_saved_conservative_pct_iter']:.2f}% T_iter)")
    print(f"  OPTIMISTIC:   {result['T_saved_optimistic_ms']:.3f}ms ({result['T_saved_optimistic_pct_iter']:.2f}% T_iter)")
    print(f"  Physical upper: {result['T_saved_upper_physical_ms']:.3f}ms ({result['T_saved_upper_physical_pct_iter']:.2f}% T_iter)")
    print(f"  Original (broken) oracle: {result['original_T_C_conservative_pct_iter']:.1f}% T_iter")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
