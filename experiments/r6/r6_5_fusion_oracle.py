#!/usr/bin/env python3
"""
R6-C: Backward-optimizer fusion oracle — memory traffic census.

Measures:
  T_optimizer: wall time of optimizer.step()
  Bytes_bwd→opt: gradient buffer bytes written by backward, read by optimizer
  Total optimizer memory traffic: param + grad + exp_avg + exp_avg_sq reads/writes

Computes the fusion oracle: if backward could directly update parameters
(bypassing intermediate grad buffers), how much traffic is avoidable?

Usage (on mx):
  CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/r6_5_fusion_oracle.py \
    --scene room --ckpt /mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_5000.pt \
    --output /mnt/storage_pool/liaoyuanjun/r6_profiling/r6_5_room_5k.json
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, argparse, math
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
from gsplat import rasterization
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
    return {
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std()), "n": len(arr),
    }


def measure_optimizer(model, cam, gt_image, ssim_fn, sh_degree, n_warmup=20, n_measure=100):
    """Measure T_optimizer and compute memory traffic."""

    opt_start = torch.cuda.Event(enable_timing=True)
    opt_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    opt_times = []
    bwd_times = []
    iter_times = []

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

    # === Memory traffic computation ===
    N = model._xyz.shape[0]
    K_active = (sh_degree + 1) ** 2
    param_sizes = {
        "xyz": N * 3 * 4,       # [N, 3] float32
        "shs": N * K_active * 3 * 4,  # [N, K, 3]
        "scaling": N * 3 * 4,
        "rotation": N * 4 * 4,
        "opacity": N * 4,       # [N] float32
    }
    total_param_bytes = sum(param_sizes.values())

    # Adam traffic per step:
    # Reads: param, grad, exp_avg, exp_avg_sq (4 × param_bytes)
    # Writes: param, exp_avg, exp_avg_sq (3 × param_bytes)
    # Total: 7 × param_bytes
    adam_traffic = 7 * total_param_bytes

    # Gradient buffer traffic (backward writes, VJP chain reads, optimizer reads .grad):
    # Backward writes: rasterizer grad buffers (v_means2d, v_conics, v_colors, v_opacities, v_means2d_abs)
    #   = N * 11 * 4 bytes
    # VJP chain reads these + writes projection/SH grad buffers
    #   = N * 10 * 4 (projection) + N * K_active * 3 * 4 (SH)
    # Parameter .grad is written by VJP chain and read by optimizer
    #   = total_param_bytes (write) + total_param_bytes (read)
    # Total grad traffic = 2 × (rasterizer + projection + SH) + 2 × param_bytes

    rasterizer_grad_bytes = N * 11 * 4  # 5 buffers
    projection_grad_bytes = N * 10 * 4  # 3 buffers
    sh_grad_bytes = N * K_active * 3 * 4  # 1 buffer
    intermediate_grad_bytes = rasterizer_grad_bytes + projection_grad_bytes + sh_grad_bytes

    # Total backward→optimizer traffic:
    # 1. Backward writes intermediate grad buffers: intermediate_grad_bytes
    # 2. VJP chain reads intermediate + writes param .grad: intermediate_grad_bytes + total_param_bytes
    # 3. Optimizer reads param .grad: total_param_bytes
    # Total grad traffic = 2 × intermediate_grad_bytes + 2 × total_param_bytes
    total_grad_traffic = 2 * intermediate_grad_bytes + 2 * total_param_bytes

    # Avoidable traffic with fusion:
    # If backward directly produces param .grad (no intermediate buffers):
    # - No intermediate grad buffer writes/reads
    # - Still need param .grad write + optimizer read
    # Avoidable = 2 × intermediate_grad_bytes
    avoidable_grad_traffic = 2 * intermediate_grad_bytes

    # T_optimizer + avoidable grad traffic opportunity
    T_opt_mean = np.mean(opt_times)
    T_bwd_mean = np.mean(bwd_times)
    T_iter_mean = np.mean(iter_times)

    # Estimate time for avoidable grad traffic at A100 bandwidth (1.55 TB/s)
    a100_bw = 1.55e12  # bytes/sec
    T_avoidable_traffic_ms = (avoidable_grad_traffic / a100_bw) * 1000

    # Fusion oracle: T_C = T_optimizer + T_avoidable_traffic (conservative)
    # This is the upper bound — assumes fusion eliminates ALL intermediate grad traffic
    # AND the optimizer can be partially overlapped
    T_C_oracle = T_opt_mean + T_avoidable_traffic_ms

    # Conservative: only the optimizer time + 50% of avoidable traffic
    # (fusion can't eliminate all traffic, some still needed)
    T_C_conservative = T_opt_mean + 0.5 * T_avoidable_traffic_ms

    return {
        "N_total": N,
        "K_active": K_active,
        "T_bwd_ms": stats_ms(bwd_times),
        "T_optimizer_ms": stats_ms(opt_times),
        "T_iter_ms": stats_ms(iter_times),
        "param_sizes_bytes": param_sizes,
        "total_param_bytes": total_param_bytes,
        "rasterizer_grad_bytes": rasterizer_grad_bytes,
        "projection_grad_bytes": projection_grad_bytes,
        "sh_grad_bytes": sh_grad_bytes,
        "intermediate_grad_bytes": intermediate_grad_bytes,
        "adam_traffic_bytes": adam_traffic,
        "total_grad_traffic_bytes": total_grad_traffic,
        "avoidable_grad_traffic_bytes": avoidable_grad_traffic,
        "T_avoidable_traffic_ms_est": T_avoidable_traffic_ms,
        "T_C_oracle_ms": T_C_oracle,
        "T_C_conservative_ms": T_C_conservative,
        "T_C_oracle_pct_iter": T_C_oracle / T_iter_mean * 100,
        "T_C_conservative_pct_iter": T_C_conservative / T_iter_mean * 100,
        "T_optimizer_pct_iter": T_opt_mean / T_iter_mean * 100,
        "T_optimizer_pct_bwd": T_opt_mean / T_bwd_mean * 100,
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

    print(f"=== R6-C Fusion Oracle: {args.scene} ===")
    dataset = GTDataset(scene=config.scene, repo_root=config.repo_root,
                        resolution=config.resolution, device="cuda", background="black")
    model = load_model_from_ckpt(args.ckpt, config, config.repo_root)
    print(f"  N={model._xyz.shape[0]}, SH={model.active_sh_degree}")

    cam, gt_image = dataset.get_item(0)
    ssim_fn = SepSSIM(device="cuda")

    result = measure_optimizer(model, cam, gt_image, ssim_fn,
                               model.active_sh_degree, args.n_warmup, args.n_measure)
    result["scene"] = args.scene
    result["checkpoint"] = args.ckpt

    print(f"\n=== Results: {args.scene} ===")
    print(f"  T_bwd: {result['T_bwd_ms']['mean']:.2f}ms")
    print(f"  T_optimizer: {result['T_optimizer_ms']['mean']:.2f}ms "
          f"({result['T_optimizer_pct_iter']:.1f}% T_iter, {result['T_optimizer_pct_bwd']:.1f}% T_bwd)")
    print(f"  T_iter: {result['T_iter_ms']['mean']:.2f}ms")
    print(f"  Intermediate grad traffic: {result['intermediate_grad_bytes']/1e6:.1f}MB")
    print(f"  Avoidable grad traffic: {result['avoidable_grad_traffic_bytes']/1e6:.1f}MB")
    print(f"  T_avoidable_traffic (est): {result['T_avoidable_traffic_ms_est']:.2f}ms")
    print(f"  T_C oracle: {result['T_C_oracle_ms']:.2f}ms ({result['T_C_oracle_pct_iter']:.1f}% T_iter)")
    print(f"  T_C conservative: {result['T_C_conservative_ms']:.2f}ms ({result['T_C_conservative_pct_iter']:.1f}% T_iter)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
