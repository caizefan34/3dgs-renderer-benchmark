#!/usr/bin/env python3
"""Phase C51 Stage 4A — CUDA Microbenchmark + Gradient Correctness

Stage 4A-1: Microbenchmark kernel latency at different K values.
Stage 4A-1: Gradient correctness for selected vs skipped Gaussians.

Uses a controlled fixed Gaussian set from SfM initialization.
Measures actual kernel time, NOT inferred speedup from skip count.
"""
import sys
import json
import time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))

import gsplat
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint


def load_room_data():
    """Load room scene SfM data and first camera."""
    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device="cuda")
    sfm_data = load_initial_checkpoint("room", repo_root, device="cuda")
    
    # Get first camera
    cam = dataset.get_camera(0)
    return sfm_data, cam


def create_model(sfm_data):
    """Create GaussianModel from SfM data."""
    N = sfm_data['xyz'].shape[0]
    model = GaussianModel(
        num_points=N,
        sh_degree=3,
        max_sh_degree=3,
    )
    # load_ply returns: opacity (logit), scales (log), rotations (raw quat)
    # init_from_sfm expects: opacity_logit, scales_log, rotations_raw
    model.init_from_sfm(
        xyz=sfm_data['xyz'].clone().float(),
        opacity_logit=sfm_data['opacity'].clone().float(),
        scales_log=sfm_data['scales'].clone().float(),
        rotations_raw=sfm_data['rotations'].clone().float(),
        shs=sfm_data['shs'].clone().float(),
    )
    model = model.cuda()
    return model


def render_with_mask(model, cam, importance_mask=None, compute_densify_grad=False):
    """Render with optional importance mask."""
    data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        importance_mask=importance_mask,
        compute_densify_grad=compute_densify_grad,
    )
    return r[0].clamp(0, 1), data


def benchmark_kernel_latency(model, cam, gt, K_values=[1.0, 0.9, 0.8, 0.7, 0.5],
                              n_warmup=10, n_measure=50, compute_densify_grad=False):
    """Benchmark backward kernel latency at different K values.
    
    Measures actual kernel time using CUDA events.
    """
    results = {}
    N = model.xyz.shape[0]
    
    for K in K_values:
        # Create mask
        if K >= 1.0:
            mask = None
            mask_name = "baseline"
        else:
            # Use random mask (consistent across iterations for timing)
            torch.manual_seed(42)
            n_keep = int(N * K)
            mask = torch.zeros(N, dtype=torch.uint8, device='cuda')
            keep_idx = torch.randperm(N)[:n_keep]
            mask[keep_idx] = 1
            mask_name = f"K{int(K*100)}"
        
        # Warmup
        for _ in range(n_warmup):
            model.zero_grad()
            if model.xyz.grad is not None:
                model.xyz.grad = None
            pred, data = render_with_mask(model, cam, mask, compute_densify_grad)
            loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * F.mse_loss(pred, gt)
            loss.backward()
        
        torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(n_measure):
            model.zero_grad()
            if model.xyz.grad is not None:
                model.xyz.grad = None
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            pred, data = render_with_mask(model, cam, mask, compute_densify_grad)
            loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * F.mse_loss(pred, gt)
            loss.backward()
            
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = np.array(times)
        results[mask_name] = {
            "K": K,
            "n_keep": int(N * K) if K < 1.0 else N,
            "mask_size": N,
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p90_ms": float(np.percentile(times, 90)),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "times": times.tolist(),
        }
        print(f"  {mask_name:>12}: mean={times.mean():.2f}ms, median={np.median(times):.2f}ms, "
              f"p90={np.percentile(times, 90):.2f}ms")
    
    # Compute speedups
    baseline_mean = results["baseline"]["mean_ms"]
    for name, r in results.items():
        r["speedup"] = baseline_mean / r["mean_ms"] if r["mean_ms"] > 0 else 0
        r["time_reduction_pct"] = (1 - r["mean_ms"] / baseline_mean) * 100 if baseline_mean > 0 else 0
    
    return results


def benchmark_forward_only(model, cam, gt, K_values=[1.0, 0.9, 0.8, 0.7, 0.5],
                            n_warmup=10, n_measure=50):
    """Benchmark forward-only (no backward) to isolate forward time."""
    results = {}
    N = model.xyz.shape[0]
    
    for K in K_values:
        if K >= 1.0:
            mask = None
            mask_name = "baseline"
        else:
            torch.manual_seed(42)
            n_keep = int(N * K)
            mask = torch.zeros(N, dtype=torch.uint8, device='cuda')
            keep_idx = torch.randperm(N)[:n_keep]
            mask[keep_idx] = 1
            mask_name = f"K{int(K*100)}"
        
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                pred, _ = render_with_mask(model, cam, mask)
        
        torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(n_measure):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            with torch.no_grad():
                pred, _ = render_with_mask(model, cam, mask)
            
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = np.array(times)
        results[mask_name] = {
            "K": K,
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
        }
        print(f"  FWD {mask_name:>12}: mean={times.mean():.2f}ms")
    
    return results


def benchmark_backward_only(model, cam, gt, K_values=[1.0, 0.9, 0.8, 0.7, 0.5],
                             n_warmup=10, n_measure=50, compute_densify_grad=False):
    """Benchmark backward-only by reusing forward outputs.
    
    This isolates the backward kernel time.
    """
    results = {}
    N = model.xyz.shape[0]
    
    for K in K_values:
        if K >= 1.0:
            mask = None
            mask_name = "baseline"
        else:
            torch.manual_seed(42)
            n_keep = int(N * K)
            mask = torch.zeros(N, dtype=torch.uint8, device='cuda')
            keep_idx = torch.randperm(N)[:n_keep]
            mask[keep_idx] = 1
            mask_name = f"K{int(K*100)}"
        
        # Do forward once to get the graph
        model.zero_grad()
        pred, data = render_with_mask(model, cam, mask, compute_densify_grad)
        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * F.mse_loss(pred, gt)
        
        # Warmup backward
        for _ in range(n_warmup):
            model.zero_grad()
            if model.xyz.grad is not None:
                model.xyz.grad = None
            pred, data = render_with_mask(model, cam, mask, compute_densify_grad)
            loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * F.mse_loss(pred, gt)
            loss.backward()
        
        torch.cuda.synchronize()
        
        # Measure full iteration (forward + backward)
        times = []
        for _ in range(n_measure):
            model.zero_grad()
            if model.xyz.grad is not None:
                model.xyz.grad = None
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            pred, data = render_with_mask(model, cam, mask, compute_densify_grad)
            loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * F.mse_loss(pred, gt)
            loss.backward()
            
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = np.array(times)
        results[mask_name] = {
            "K": K,
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "p90_ms": float(np.percentile(times, 90)),
        }
        print(f"  FULL {mask_name:>12}: mean={times.mean():.2f}ms")
    
    return results


def gradient_correctness_test(model, cam, gt, K=0.8, compute_densify_grad=False):
    """Test gradient correctness for selected vs skipped Gaussians.
    
    For selected Gaussians (mask=1): gradient should match baseline exactly.
    For skipped Gaussians (mask=0): gradient should be exactly zero (B1/B3).
    For B2 mode: v_means2d should be correct for all, other grads zero for skipped.
    """
    N = model.xyz.shape[0]
    
    # Create mask: keep top 80% randomly
    torch.manual_seed(42)
    n_keep = int(N * K)
    mask = torch.zeros(N, dtype=torch.uint8, device='cuda')
    keep_idx = torch.randperm(N)[:n_keep]
    mask[keep_idx] = 1
    
    # Baseline: full backward (no mask)
    model.zero_grad()
    pred_base, data_base = render_with_mask(model, cam, None)
    loss_base = 0.8 * F.l1_loss(pred_base, gt) + 0.2 * F.mse_loss(pred_base, gt)
    loss_base.backward()
    
    grads_base = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads_base[name] = p.grad.clone()
    
    # Sparse: with mask
    model.zero_grad()
    pred_sparse, data_sparse = render_with_mask(model, cam, mask, compute_densify_grad)
    loss_sparse = 0.8 * F.l1_loss(pred_sparse, gt) + 0.2 * F.mse_loss(pred_sparse, gt)
    loss_sparse.backward()
    
    grads_sparse = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads_sparse[name] = p.grad.clone()
    
    # Compare
    results = {}
    mask_bool = mask.bool()
    selected = mask_bool  # mask=1
    skipped = ~mask_bool  # mask=0
    
    for name in grads_base:
        if name not in grads_sparse:
            continue
        g_base = grads_base[name]
        g_sparse = grads_sparse[name]
        
        # Use boolean mask directly (shape [N]) for indexing [N, D] tensors
        # selected: mask=1 (compute), skipped: mask=0 (skip)
        
        # Selected Gaussian gradients (should match baseline)
        g_base_sel = g_base[selected]  # [n_sel, D]
        g_sparse_sel = g_sparse[selected]  # [n_sel, D]
        
        if g_base_sel.numel() > 0:
            base_norm = g_base_sel.norm()
            if base_norm > 1e-12:
                rel_l2_sel = (g_sparse_sel - g_base_sel).norm() / base_norm
                cosine_sel = torch.dot(g_sparse_sel.flatten(), g_base_sel.flatten()) / (
                    g_sparse_sel.norm() * base_norm + 1e-12)
            else:
                rel_l2_sel = torch.tensor(0.0)
                cosine_sel = torch.tensor(1.0)
            max_abs_sel = (g_sparse_sel - g_base_sel).abs().max()
        else:
            rel_l2_sel = torch.tensor(0.0)
            cosine_sel = torch.tensor(1.0)
            max_abs_sel = torch.tensor(0.0)
        
        # Skipped Gaussian gradients (should be zero for B1/B3)
        g_sparse_skip = g_sparse[skipped]  # [n_skip, D]
        skip_max = g_sparse_skip.abs().max() if g_sparse_skip.numel() > 0 else torch.tensor(0.0)
        skip_norm = g_sparse_skip.norm() if g_sparse_skip.numel() > 0 else torch.tensor(0.0)
        
        # For B2: xyz grad should be non-zero for skipped (densification path)
        # For B1/B3: xyz grad should be zero for skipped
        results[name] = {
            "selected": {
                "cosine": float(cosine_sel),
                "rel_l2": float(rel_l2_sel),
                "max_abs_diff": float(max_abs_sel),
                "n_elements": g_base_sel.numel(),
            },
            "skipped": {
                "max_abs": float(skip_max),
                "norm": float(skip_norm),
                "n_elements": g_sparse_skip.numel(),
                "is_zero": float(skip_max) < 1e-10,
            }
        }
        
        mode = "B2" if compute_densify_grad else "B1/B3"
        print(f"  {name:>12} [{mode}]: sel_cos={cosine_sel:.6f}, sel_rel_l2={rel_l2_sel:.6f}, "
              f"skip_max={skip_max:.8f}, skip_norm={skip_norm:.8f}")
    
    # Forward correctness
    fwd_diff = (pred_base - pred_sparse).abs().max().item()
    results["forward"] = {"max_diff": fwd_diff}
    print(f"  Forward max diff: {fwd_diff:.10f}")
    
    results["mask_info"] = {
        "K": K,
        "n_selected": int(n_keep),
        "n_skipped": N - n_keep,
        "compute_densify_grad": compute_densify_grad,
    }
    
    return results


def main():
    print("=" * 70)
    print("Phase C51 Stage 4A — CUDA Microbenchmark + Gradient Correctness")
    print("=" * 70)
    
    # Load data
    print("\nLoading room scene data...")
    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device="cuda")
    sfm_data = load_initial_checkpoint("room", repo_root, device="cuda")
    cam = dataset.get_camera(0)
    gt = dataset.get_gt_image(0)  # [H, W, 3] float32 on cuda
    model = create_model(sfm_data)
    N = model.xyz.shape[0]
    print(f"  N={N}, image={cam.image_width}x{cam.image_height}")
    print(f"  GT shape: {gt.shape}")
    
    all_results = {}
    
    # === Stage 4A-1: Microbenchmark ===
    print("\n" + "=" * 70)
    print("1. Forward-Only Latency (should be identical for all K)")
    print("=" * 70)
    fwd_times = benchmark_forward_only(model, cam, gt)
    all_results["forward_only"] = fwd_times
    
    print("\n" + "=" * 70)
    print("2. Full Iteration (Forward + Backward) — B1/B3 mode (skip all grad)")
    print("=" * 70)
    full_b1 = benchmark_kernel_latency(model, cam, gt, compute_densify_grad=False)
    all_results["full_b1"] = full_b1
    
    print("\n" + "=" * 70)
    print("3. Full Iteration — B2 mode (compute v_means2d for masked)")
    print("=" * 70)
    full_b2 = benchmark_kernel_latency(model, cam, gt, compute_densify_grad=True)
    all_results["full_b2"] = full_b2
    
    # === Gradient Correctness ===
    print("\n" + "=" * 70)
    print("4. Gradient Correctness — B1/B3 mode (K=80%)")
    print("=" * 70)
    corr_b1 = gradient_correctness_test(model, cam, gt, K=0.8, compute_densify_grad=False)
    all_results["correctness_b1"] = corr_b1
    
    print("\n" + "=" * 70)
    print("5. Gradient Correctness — B2 mode (K=80%)")
    print("=" * 70)
    corr_b2 = gradient_correctness_test(model, cam, gt, K=0.8, compute_densify_grad=True)
    all_results["correctness_b2"] = corr_b2
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print("\nForward-only (should be identical):")
    for name, r in fwd_times.items():
        print(f"  {name:>12}: {r['mean_ms']:.2f} ms")
    
    print("\nFull iteration B1/B3 (skip all gradient):")
    base = full_b1["baseline"]["mean_ms"]
    for name, r in full_b1.items():
        print(f"  {name:>12}: {r['mean_ms']:.2f} ms  (speedup={r['speedup']:.3f}x, "
              f"reduction={r['time_reduction_pct']:.1f}%)")
    
    print("\nFull iteration B2 (compute v_means2d only for masked):")
    base2 = full_b2["baseline"]["mean_ms"]
    for name, r in full_b2.items():
        print(f"  {name:>12}: {r['mean_ms']:.2f} ms  (speedup={r['speedup']:.3f}x, "
              f"reduction={r['time_reduction_pct']:.1f}%)")
    
    print("\nGradient Correctness B1/B3:")
    for name, r in corr_b1.items():
        if name in ("forward", "mask_info"):
            continue
        print(f"  {name:>12}: sel_cos={r['selected']['cosine']:.6f}, "
              f"skip_zero={r['skipped']['is_zero']}")
    
    print("\nGradient Correctness B2:")
    for name, r in corr_b2.items():
        if name in ("forward", "mask_info"):
            continue
        print(f"  {name:>12}: sel_cos={r['selected']['cosine']:.6f}, "
              f"skip_max={r['skipped']['max_abs']:.8f}")
    
    # Save results
    out_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4a")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kernel_microbenchmark.json"
    
    # Convert numpy types
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (list,)):
            return [clean(v) for v in obj]
        return obj
    
    with open(out_path, 'w') as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
