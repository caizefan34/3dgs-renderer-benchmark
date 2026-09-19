#!/usr/bin/env python3
"""
R6-A direct kernel instrumentation runner.

Measures the ACTUAL number of warp-leader atomicAdd execution sets in the
rasterize_to_pixels_3dgs_bwd kernel by reading the __device__ debug counter
patched into the gsplat source.

This is DIRECT DEBUG INSTRUMENTATION (evidence level 2) — higher quality
than the pixel-level simulation (which is an estimate) and the footprint
method (which is a metadata-derived oracle).

For each workload:
  1. Load checkpoint
  2. Forward pass → extract n_isects from metadata
  3. Reset debug counter
  4. Backward pass (the patched kernel increments the counter)
  5. Read debug counter → actual_atomic
  6. R_atomic_direct = actual_atomic / n_isects
  7. Conservative E2E = 5 * (R_atomic_direct - 1) * n_isects / 30e9 * 1000 * 0.70 / T_iter * 100

Gate conditions:
  - R_atomic_direct >= 2 (direct reduction potential)
  - Conservative E2E >= 5%
  - At least 2 workloads pass

Usage (on mx):
  CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 \
    PYTHONPATH=/tmp/r6a_patched TORCH_CUDA_ARCH_LIST=8.0 FAST_COMPILE=1 \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/r6_a/run_debug_atomic.py \
    --scenes room bicycle garden \
    --output /mnt/storage_pool/liaoyuanjun/r6_profiling/r6_a_direct_kernel.json
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/liaoyuanjun/3dgs-renderer-benchmark"))
if not REPO_ROOT.exists():
    # Fall back to path relative to this script (for in-repo execution)
    script_root = Path(__file__).resolve().parent
    for _ in range(5):
        if (script_root / "baseline" / "reference_v1").is_dir():
            REPO_ROOT = script_root
            break
        script_root = script_root.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
sys.path.insert(0, str(REPO_ROOT / "src"))
# Patched gsplat must be first in path (set via PYTHONPATH)
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat.cuda._backend import _C
from dataset import GTDataset
from trainer import SepSSIM, render_with_meta

# Debug counter functions from the patched kernel
r6a_reset = _C.r6a_reset_debug_counter
r6a_get = _C.r6a_get_debug_counter


CKPT_BASE = "/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts"
STAGES = [5000, 15000, 30000]


def load_model(ckpt_path, config):
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


def measure_workload(scene, stage, config, n_warmup=10, n_measure=5):
    """Run instrumented backward and measure actual atomic count."""
    ckpt_path = f"{CKPT_BASE}/{scene}/checkpoints/iter_{stage}.pt"
    config.scene = scene
    config.repo_root = str(REPO_ROOT)

    dataset = GTDataset(scene=scene, repo_root=str(REPO_ROOT),
                        resolution=config.resolution, device="cuda",
                        background="black")
    model = load_model(ckpt_path, config)
    cam, gt_image = dataset.get_item(0)
    ssim_fn = SepSSIM(device="cuda")
    sh_degree = model.active_sh_degree

    # Warmup
    for _ in range(n_warmup):
        model.optimizer.zero_grad(set_to_none=True)
        image, meta, means2d = render_with_meta(model, cam, sh_degree)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - 0.2) * L1 + 0.2 * dssim
        loss.backward()
        torch.cuda.synchronize()

    # Get n_isects from metadata
    flatten_ids = meta["flatten_ids"]
    n_isects = int(flatten_ids.shape[0])
    N_total = model._xyz.shape[0]

    # Measure T_iter with CUDA events
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    iter_times = []
    atomic_counts = []

    for _ in range(n_measure):
        model.optimizer.zero_grad(set_to_none=True)

        iter_start.record()
        image, meta, means2d = render_with_meta(model, cam, sh_degree)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - 0.2) * L1 + 0.2 * dssim

        # Reset counter BEFORE backward
        r6a_reset()

        loss.backward()
        torch.cuda.synchronize()

        # Read counter AFTER backward
        actual_atomic = r6a_get()

        iter_end.record()
        torch.cuda.synchronize()
        iter_ms = iter_start.elapsed_time(iter_end)

        iter_times.append(iter_ms)
        atomic_counts.append(actual_atomic)

        # Update n_isects from this iteration's metadata
        flatten_ids = meta["flatten_ids"]
        n_isects = int(flatten_ids.shape[0])

    # Compute statistics
    actual_atomic_mean = float(np.mean(atomic_counts))
    actual_atomic_std = float(np.std(atomic_counts))
    T_iter_mean = float(np.mean(iter_times))

    R_atomic_direct = actual_atomic_mean / max(n_isects, 1)

    # Conservative E2E oracle (hardware throughput model, level 4 evidence)
    # T_atomic = 5 * R_atomic * n_isects / 30e9 * 1000  (ms, 5 cache lines, 30 G/s)
    # T_atomic_ideal = 5 * n_isects / 30e9 * 1000  (ms, block-aggregated)
    # Savings = (T_atomic - T_atomic_ideal) * 0.70  (30% __syncthreads overhead)
    # E2E = Savings / T_iter * 100
    T_atomic = 5.0 * R_atomic_direct * n_isects / 30e9 * 1000.0  # ms
    T_atomic_ideal = 5.0 * n_isects / 30e9 * 1000.0  # ms
    savings_ms = (T_atomic - T_atomic_ideal) * 0.70
    e2e_pct = savings_ms / max(T_iter_mean, 1e-6) * 100.0

    # Gate conditions
    gate_r = R_atomic_direct >= 2.0
    gate_e2e = e2e_pct >= 5.0
    gate_pass = gate_r and gate_e2e

    result = {
        "scene": scene,
        "stage": stage,
        "N_total": N_total,
        "n_isects": n_isects,
        "actual_atomic_mean": actual_atomic_mean,
        "actual_atomic_std": actual_atomic_std,
        "actual_atomic_samples": atomic_counts,
        "R_atomic_direct": R_atomic_direct,
        "T_iter_ms": T_iter_mean,
        "T_iter_samples": iter_times,
        "T_atomic_ms": T_atomic,
        "T_atomic_ideal_ms": T_atomic_ideal,
        "savings_ms": savings_ms,
        "e2e_pct": e2e_pct,
        "gate_r_atomic_ge2": gate_r,
        "gate_e2e_ge5": gate_e2e,
        "gate_pass": gate_pass,
        "evidence_level": "direct_debug_instrumentation",
        "alpha_threshold_kernel": "1.0/255.0",  # ALPHA_THRESHOLD from Common.h
    }

    print(f"  {scene} {stage:>5d}: n_isects={n_isects:>10d}, "
          f"actual_atomic={actual_atomic_mean:>12.1f}±{actual_atomic_std:.1f}, "
          f"R_atomic={R_atomic_direct:.3f}, "
          f"E2E={e2e_pct:.2f}%, "
          f"{'PASS' if gate_pass else 'FAIL'}")

    # Cleanup
    del model, dataset, cam, gt_image, ssim_fn
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["room", "bicycle", "garden"])
    parser.add_argument("--stages", nargs="+", type=int, default=[5000, 15000, 30000])
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-measure", type=int, default=5)
    args = parser.parse_args()

    config = ReferenceV1Config()
    config.resolution = args.resolution
    config.seed = 0

    print("=" * 80)
    print("R6-A Direct Kernel Instrumentation (debug atomic counter)")
    print("=" * 80)
    print(f"  Evidence level: DIRECT DEBUG INSTRUMENTATION (level 2)")
    print(f"  Counter: __device__ uint64_t incremented per warp-leader atomic set")
    print(f"  Kernel threshold: ALPHA_THRESHOLD = 1/255 ≈ 0.00392")
    print()

    all_results = []
    for scene in args.scenes:
        for stage in args.stages:
            try:
                result = measure_workload(scene, stage, config,
                                          args.n_warmup, args.n_measure)
                all_results.append(result)
            except Exception as e:
                print(f"  {scene} {stage}: ERROR — {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "scene": scene, "stage": stage, "error": str(e)
                })

    # Summary
    valid = [r for r in all_results if "error" not in r]
    n_pass = sum(1 for r in valid if r["gate_pass"])
    n_r_pass = sum(1 for r in valid if r["gate_r_atomic_ge2"])
    n_e2e_pass = sum(1 for r in valid if r["gate_e2e_ge5"])

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Workloads measured: {len(valid)}/{len(all_results)}")
    print(f"  R_atomic_direct range: {min(r['R_atomic_direct'] for r in valid):.3f} — "
          f"{max(r['R_atomic_direct'] for r in valid):.3f}")
    print(f"  E2E range: {min(r['e2e_pct'] for r in valid):.2f}% — "
          f"{max(r['e2e_pct'] for r in valid):.2f}%")
    print(f"  Gate R_atomic>=2: {n_r_pass}/{len(valid)} pass")
    print(f"  Gate E2E>=5%: {n_e2e_pass}/{len(valid)} pass")
    print(f"  Gate overall (both): {n_pass}/{len(valid)} pass")
    print()

    if n_pass >= 2:
        print("  VERDICT: ADVANCE — at least 2 workloads pass both gates")
    elif n_r_pass >= 2 and n_e2e_pass == 0:
        print("  VERDICT: DEFER — R_atomic passes but E2E too small")
    elif n_r_pass < 2:
        print("  VERDICT: DROP — R_atomic < 2 for most workloads")
    else:
        print("  VERDICT: DEFER — insufficient workloads pass")

    # Save
    output = {
        "method": "direct_kernel_instrumentation",
        "evidence_level": "direct_debug_instrumentation",
        "description": "Actual warp-leader atomicAdd execution count from __device__ counter",
        "kernel_alpha_threshold": "1/255",
        "hardware_model_atomic_throughput_gps": 30.0,
        "syncthreads_overhead_pct": 30,
        "results": all_results,
        "summary": {
            "n_valid": len(valid),
            "n_pass": n_pass,
            "n_r_pass": n_r_pass,
            "n_e2e_pass": n_e2e_pass,
            "R_atomic_range": [min(r["R_atomic_direct"] for r in valid),
                               max(r["R_atomic_direct"] for r in valid)],
            "e2e_range": [min(r["e2e_pct"] for r in valid),
                          max(r["e2e_pct"] for r in valid)],
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
