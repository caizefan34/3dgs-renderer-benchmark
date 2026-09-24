#!/usr/bin/env python3
"""
C40-0: Training Kernel Bottleneck Profiling.

Profiles a real training iteration and breaks down time into:
  Forward:  projection, intersection, sorting, rasterization
  Backward: rasterization_bwd, projection_bwd
  Optim:    Adam update
  Loss:     L1 + D-SSIM
  Other:    misc PyTorch ops

Uses two complementary methods:
  1. CUDA events around high-level Python operations (forward, backward, optimizer, loss)
  2. torch.profiler for kernel-level breakdown (maps kernel names to stages)
"""
import json, math, os, time, sys, re
from pathlib import Path
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


# ─── Kernel name → stage categorization ───
def categorize_kernel(name: str) -> str:
    """Map a CUDA kernel name to a training stage."""
    nl = name.lower()

    # gsplat forward kernels
    if "fully_fused_projection" in nl and "bwd" not in nl:
        return "projection_fwd"
    if "isect_tiles" in nl:
        return "intersection"
    if "isect_offset" in nl:
        return "intersection"
    if "rasterize_to_pixels" in nl and "bwd" not in nl:
        return "rasterize_fwd"
    if "spherical_harmonics" in nl and "bwd" not in nl:
        return "projection_fwd"  # SH evaluation is part of forward projection

    # gsplat backward kernels
    if "rasterize_to_pixels" in nl and "bwd" in nl:
        return "rasterize_bwd"
    if "fully_fused_projection" in nl and "bwd" in nl:
        return "projection_bwd"
    if "spherical_harmonics" in nl and "bwd" in nl:
        return "projection_bwd"

    # Sorting (CUB radix sort)
    if "radixsort" in nl or "radix_sort" in nl or "segmentedsort" in nl:
        return "sorting"
    if "cub" in nl and "sort" in nl:
        return "sorting"

    # Adam optimizer
    if "multi_tensor_apply" in nl:
        return "adam"
    if "adam" in nl:
        return "adam"

    # Loss computation (D-SSIM uses convolution)
    if "cutlass" in nl or "cudnn" in nl:
        return "loss_dssim"
    if "tensorTransform" in nl:
        return "loss_dssim"
    if "dgrad" in nl or "wgrad" in nl or "fgrad" in nl:
        return "loss_dssim"
    if "convolution" in nl or "conv_" in nl:
        return "loss_dssim"

    # Memory ops
    if "memcpy" in nl or "memset" in nl:
        return "memory"
    if "fill" in nl and "elementwise" in nl:
        return "memory"

    # Generic elementwise (activations, loss L1, etc.)
    if "elementwise" in nl or "pointwise" in nl or "binary" in nl:
        return "elementwise_misc"

    # Reductions (norm, sum)
    if "reduce" in nl or "reduction" in nl:
        return "reduction"

    return "other"


def main():
    print("=" * 72)
    print("C40-0: Training Kernel Bottleneck Profiling")
    print("=" * 72)

    device = "cuda"
    torch.manual_seed(42)

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
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=device)
    cam0 = dataset.get_camera(0)
    gt0 = dataset.get_gt_image(0)
    img_w, img_h = cam0.image_width, cam0.image_height
    print(f"  Camera 0: {img_w}x{img_h}")

    # Load model (same as C38/C39)
    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    print(f"\n  Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=device)
    N = model.xyz.shape[0]
    print(f"  Model: {N:,} Gaussians, SH degree: {model.sh_degree}")

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * SPATIAL_LR_SCALE, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    # ─── Warmup ───
    WARMUP = 30
    N_PROFILE = 20  # profiled iterations

    print(f"\n  Warmup ({WARMUP} iters)...")
    for i in range(WARMUP):
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        loss = combined_loss(rendered[0].clamp(0, 1), gt0, lambda_dssim=0.2)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # ═══════════════════════════════════════════════════
    # METHOD 1: CUDA events around high-level operations
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Method 1: CUDA Events — {N_PROFILE} iterations")
    print(f"{'='*60}")

    ev_fwd_start = torch.cuda.Event(enable_timing=True)
    ev_fwd_end = torch.cuda.Event(enable_timing=True)
    ev_bwd_start = torch.cuda.Event(enable_timing=True)
    ev_bwd_end = torch.cuda.Event(enable_timing=True)
    ev_opt_start = torch.cuda.Event(enable_timing=True)
    ev_opt_end = torch.cuda.Event(enable_timing=True)
    ev_loss_start = torch.cuda.Event(enable_timing=True)
    ev_loss_end = torch.cuda.Event(enable_timing=True)
    ev_total_start = torch.cuda.Event(enable_timing=True)
    ev_total_end = torch.cuda.Event(enable_timing=True)

    event_timings = []

    for iter_idx in range(1, N_PROFILE + 1):
        ev_total_start.record()

        # Forward (render)
        data = model.forward()
        ev_fwd_start.record()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        ev_fwd_end.record()
        rendered_img = rendered[0].clamp(0, 1)

        # Loss
        ev_loss_start.record()
        loss_dict = combined_loss(rendered_img, gt0, lambda_dssim=0.2)
        loss = loss_dict["loss"]
        ev_loss_end.record()

        # Backward
        ev_bwd_start.record()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        ev_bwd_end.record()

        # Grad clip + optimizer
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        ev_opt_start.record()
        optimizer.step()
        torch.cuda.synchronize()
        ev_opt_end.record()

        ev_total_end.record()
        torch.cuda.synchronize()

        fwd_ms = ev_fwd_start.elapsed_time(ev_fwd_end)
        loss_ms = ev_loss_start.elapsed_time(ev_loss_end)
        bwd_ms = ev_bwd_start.elapsed_time(ev_bwd_end)
        opt_ms = ev_opt_start.elapsed_time(ev_opt_end)
        total_ms = ev_total_start.elapsed_time(ev_total_end)

        event_timings.append({
            "iter": iter_idx,
            "fwd_ms": fwd_ms,
            "loss_ms": loss_ms,
            "bwd_ms": bwd_ms,
            "opt_ms": opt_ms,
            "total_ms": total_ms,
        })

        if iter_idx % 5 == 0:
            print(f"  iter={iter_idx:3d}  fwd={fwd_ms:.2f}ms  loss={loss_ms:.2f}ms  "
                  f"bwd={bwd_ms:.2f}ms  opt={opt_ms:.2f}ms  total={total_ms:.2f}ms")

    # ═══════════════════════════════════════════════════
    # METHOD 2: torch.profiler for kernel-level breakdown
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Method 2: torch.profiler — 10 iterations")
    print(f"{'='*60}")

    # Warmup profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=10, repeat=1),
        on_trace_ready=lambda p: None,
    ) as prof:
        for iter_idx in range(15):
            data = model.forward()
            rendered, _, info = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
                width=img_w, height=img_h,
                tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
                radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
            )
            rendered_img = rendered[0].clamp(0, 1)
            loss = combined_loss(rendered_img, gt0, lambda_dssim=0.2)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            prof.step()

    # Extract kernel-level data from profiler via Chrome trace export
    print("\n  Extracting kernel-level data from profiler...")

    # Export to Chrome trace JSON and parse
    trace_path = Path("results/phase-c31/c40_trace.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    with open(trace_path) as f:
        trace = json.load(f)

    # Parse trace events — look for CUDA kernel events
    kernel_events = []
    for evt in trace.get("traceEvents", []):
        if evt.get("cat") == "kernel" and evt.get("dur", 0) > 0:
            kernel_name = evt.get("name", "")
            duration_us = evt.get("dur", 0)  # microseconds
            kernel_events.append({
                "name": kernel_name,
                "duration_us": duration_us,
                "category": categorize_kernel(kernel_name),
            })

    print(f"  Total CUDA kernel events: {len(kernel_events)}")

    # Aggregate by category
    category_totals = defaultdict(lambda: {"total_us": 0, "count": 0, "kernels": defaultdict(lambda: {"total_us": 0, "count": 0})})

    for ke in kernel_events:
        cat = ke["category"]
        category_totals[cat]["total_us"] += ke["duration_us"]
        category_totals[cat]["count"] += 1
        category_totals[cat]["kernels"][ke["name"]]["total_us"] += ke["duration_us"]
        category_totals[cat]["kernels"][ke["name"]]["count"] += 1

    # The profiler ran 10 active iterations, so divide by 10
    N_PROF_ITERS = 10

    print(f"\n  Per-iteration kernel breakdown ({N_PROF_ITERS} profiled iters):")
    print(f"  {'Category':<25s}  {'Time/iter (ms)':>14s}  {'% of total':>10s}  {'Kernels/iter':>12s}")
    print(f"  {'-'*25}  {'-'*14}  {'-'*10}  {'-'*12}")

    total_all_us = sum(v["total_us"] for v in category_totals.values())
    breakdown = {}

    for cat in sorted(category_totals.keys(), key=lambda k: category_totals[k]["total_us"], reverse=True):
        v = category_totals[cat]
        per_iter_ms = v["total_us"] / N_PROF_ITERS / 1000
        pct = v["total_us"] / total_all_us * 100
        kernels_per_iter = v["count"] / N_PROF_ITERS
        print(f"  {cat:<25s}  {per_iter_ms:>14.3f}  {pct:>9.1f}%  {kernels_per_iter:>12.1f}")
        breakdown[cat] = {
            "per_iter_ms": per_iter_ms,
            "pct": pct,
            "kernels_per_iter": kernels_per_iter,
            "total_us": v["total_us"],
            "count": v["count"],
        }

    # Print top kernels per category
    print(f"\n  Top kernels per category:")
    for cat in sorted(category_totals.keys(), key=lambda k: category_totals[k]["total_us"], reverse=True):
        v = category_totals[cat]
        top_kernels = sorted(v["kernels"].items(), key=lambda x: x[1]["total_us"], reverse=True)[:3]
        print(f"\n  [{cat}]")
        for kname, kdata in top_kernels:
            short_name = kname[:100] + "..." if len(kname) > 100 else kname
            calls_per_iter = kdata["count"] / N_PROF_ITERS
            print(f"    {short_name}")
            print(f"      {kdata['total_us']/N_PROF_ITERS/1000:.3f} ms/iter  ({calls_per_iter:.0f} calls/iter)")

    # ═══════════════════════════════════════════════════
    # Analysis & Decision
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("C40-0 TRAINING BREAKDOWN SUMMARY")
    print(f"{'='*72}")

    # Method 1 summary
    fwd_arr = np.array([t["fwd_ms"] for t in event_timings])
    loss_arr = np.array([t["loss_ms"] for t in event_timings])
    bwd_arr = np.array([t["bwd_ms"] for t in event_timings])
    opt_arr = np.array([t["opt_ms"] for t in event_timings])
    total_arr = np.array([t["total_ms"] for t in event_timings])

    print(f"\n  Method 1: CUDA Events (N={N_PROFILE} iters)")
    print(f"  {'Stage':<15s}  {'Mean (ms)':>10s}  {'Median':>10s}  {'P90':>10s}  {'% of total':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    mean_total = total_arr.mean()
    for name, arr in [("Forward", fwd_arr), ("Loss", loss_arr), ("Backward", bwd_arr), ("Optimizer", opt_arr), ("Total", total_arr)]:
        pct = arr.mean() / mean_total * 100 if name != "Total" else 100
        print(f"  {name:<15s}  {arr.mean():>10.2f}  {np.median(arr):>10.2f}  {np.percentile(arr,90):>10.2f}  {pct:>9.1f}%")

    # Method 2 summary (from profiler)
    print(f"\n  Method 2: torch.profiler kernel breakdown (N={N_PROF_ITERS} iters)")
    prof_total = total_all_us / N_PROF_ITERS / 1000
    print(f"  Total GPU kernel time: {prof_total:.2f} ms/iter")
    for cat in sorted(breakdown.keys(), key=lambda k: breakdown[k]["per_iter_ms"], reverse=True):
        v = breakdown[cat]
        print(f"    {cat:<25s}  {v['per_iter_ms']:>8.3f} ms  {v['pct']:>6.1f}%  ({v['kernels_per_iter']:.0f} kernels/iter)")

    # Decision gate
    print(f"\n{'='*72}")
    print("DECISION GATE")
    print(f"{'='*72}")

    fwd_pct = fwd_arr.mean() / mean_total * 100
    bwd_pct = bwd_arr.mean() / mean_total * 100

    # From profiler breakdown
    proj_fwd_pct = breakdown.get("projection_fwd", {}).get("pct", 0)
    intersect_pct = breakdown.get("intersection", {}).get("pct", 0)
    sort_pct = breakdown.get("sorting", {}).get("pct", 0)
    rast_fwd_pct = breakdown.get("rasterize_fwd", {}).get("pct", 0)
    rast_bwd_pct = breakdown.get("rasterize_bwd", {}).get("pct", 0)
    proj_bwd_pct = breakdown.get("projection_bwd", {}).get("pct", 0)
    adam_pct = breakdown.get("adam", {}).get("pct", 0)
    loss_pct = breakdown.get("loss_dssim", {}).get("pct", 0)

    print(f"\n  Backward (events): {bwd_pct:.1f}% of total — {'>50%' if bwd_pct > 50 else '<50%'}")
    print(f"  Projection (profiler): {proj_fwd_pct:.1f}% — {'>30%' if proj_fwd_pct > 30 else '<30%'}")
    print(f"  Intersection (profiler): {intersect_pct:.1f}% — {'>30%' if intersect_pct > 30 else '<30%'}")
    print(f"  Sorting (profiler): {sort_pct:.1f}%")
    print(f"  Rasterize fwd (profiler): {rast_fwd_pct:.1f}%")
    print(f"  Rasterize bwd (profiler): {rast_bwd_pct:.1f}%")
    print(f"  Projection bwd (profiler): {proj_bwd_pct:.1f}%")
    print(f"  Adam (profiler): {adam_pct:.1f}%")
    print(f"  Loss D-SSIM (profiler): {loss_pct:.1f}%")

    # Determine dominant stage
    stages = [
        ("backward (rasterize_bwd + projection_bwd)", rast_bwd_pct + proj_bwd_pct),
        ("projection_fwd", proj_fwd_pct),
        ("intersection", intersect_pct),
        ("rasterize_fwd", rast_fwd_pct),
        ("sorting", sort_pct),
        ("adam", adam_pct),
        ("loss_dssim", loss_pct),
    ]
    dominant_stage, dominant_pct = max(stages, key=lambda x: x[1])

    print(f"\n  Dominant stage: {dominant_stage} ({dominant_pct:.1f}%)")

    if rast_bwd_pct + proj_bwd_pct > 50:
        print(f"\n  → Backward > 50%: investigate backward kernel optimization")
    elif proj_fwd_pct > 30:
        print(f"\n  → Projection > 30%: investigate projection optimization")
    elif intersect_pct > 30:
        print(f"\n  → Intersection > 30%: investigate new intersection algorithm")
    else:
        print(f"\n  → No single stage dominant: perform roofline/memory analysis")

    # ═══════════════════════════════════════════════════
    # Save output
    # ═══════════════════════════════════════════════════
    output = {
        "config": {
            "scene": "room",
            "resolution": "1080p",
            "n_gaussians": N,
            "sh_degree": SH_DEGREE,
            "tile_size": TILE_SIZE,
            "packed": PACKED,
            "densification": "disabled",
            "warmup": WARMUP,
            "n_profile_iters": N_PROFILE,
        },
        "method1_cuda_events": {
            "per_iter": [
                {k: float(v) for k, v in t.items()} for t in event_timings
            ],
            "summary": {
                "fwd_ms": {"mean": float(fwd_arr.mean()), "median": float(np.median(fwd_arr)), "p90": float(np.percentile(fwd_arr, 90))},
                "loss_ms": {"mean": float(loss_arr.mean()), "median": float(np.median(loss_arr)), "p90": float(np.percentile(loss_arr, 90))},
                "bwd_ms": {"mean": float(bwd_arr.mean()), "median": float(np.median(bwd_arr)), "p90": float(np.percentile(bwd_arr, 90))},
                "opt_ms": {"mean": float(opt_arr.mean()), "median": float(np.median(opt_arr)), "p90": float(np.percentile(opt_arr, 90))},
                "total_ms": {"mean": float(total_arr.mean()), "median": float(np.median(total_arr)), "p90": float(np.percentile(total_arr, 90))},
            },
        },
        "method2_profiler_breakdown": {
            cat: {
                "per_iter_ms": float(v["per_iter_ms"]),
                "pct": float(v["pct"]),
                "kernels_per_iter": float(v["kernels_per_iter"]),
                "count": int(v["count"]),
            }
            for cat, v in breakdown.items()
        },
        "decision_gate": {
            "backward_pct": float(rast_bwd_pct + proj_bwd_pct),
            "projection_fwd_pct": float(proj_fwd_pct),
            "intersection_pct": float(intersect_pct),
            "sorting_pct": float(sort_pct),
            "rasterize_fwd_pct": float(rast_fwd_pct),
            "rasterize_bwd_pct": float(rast_bwd_pct),
            "projection_bwd_pct": float(proj_bwd_pct),
            "adam_pct": float(adam_pct),
            "loss_dssim_pct": float(loss_pct),
            "dominant_stage": dominant_stage,
            "dominant_pct": float(dominant_pct),
        },
    }

    save_path = Path("results/phase-c31/c40_training_breakdown.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
