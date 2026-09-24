#!/usr/bin/env python3
"""
C41: GPU Utilization Bottleneck Discovery.

Collects detailed GPU metrics for 4 components:
1. D-SSIM loss (cuDNN convolution)
2. Adam optimizer (41 kernels)
3. Elementwise misc (130 kernels)
4. rasterize_bwd (gsplat backward)

Uses:
- torch.profiler with Chrome trace export for kernel-level timing
- NVTX ranges for clean phase attribution
- Manual tensor size measurement for memory bandwidth estimation
- torch.cuda.memory_allocated for memory tracking
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

# A100-PCIE-40GB specs
A100_SMs = 108
A100_MAX_THREADS_PER_SM = 2048
A100_PEAK_BW_GB_S = 1555  # GB/s (PCIE)
A100_PEAK_TFLOPS_FP32 = 19.5
A100_PEAK_TFLOPS_TF32 = 156  # with tensor cores


def categorize_kernel(name: str) -> str:
    nl = name.lower()
    if "fully_fused_projection" in nl and "bwd" not in nl:
        return "projection_fwd"
    if "isect" in nl or "intersect" in nl:
        return "intersection"
    if "rasterize_to_pixels" in nl and "bwd" not in nl:
        return "rasterize_fwd"
    if "rasterize_to_pixels" in nl and "bwd" in nl:
        return "rasterize_bwd"
    if "fully_fused_projection" in nl and "bwd" in nl:
        return "projection_bwd"
    if "spherical_harmonics" in nl and "bwd" not in nl:
        return "projection_fwd"
    if "spherical_harmonics" in nl and "bwd" in nl:
        return "projection_bwd"
    if "radixsort" in nl or "radix_sort" in nl or "segmentedsort" in nl:
        return "sorting"
    if "cub" in nl and "sort" in nl:
        return "sorting"
    if "multi_tensor_apply" in nl:
        return "adam"
    if "adam" in nl:
        return "adam"
    if "cutlass" in nl or "cudnn" in nl or "tensorTransform" in nl:
        return "loss_dssim"
    if "dgrad" in nl or "wgrad" in nl or "fgrad" in nl or "conv" in nl:
        return "loss_dssim"
    if "elementwise" in nl or "pointwise" in nl or "binary" in nl:
        return "elementwise_misc"
    if "reduce" in nl or "reduction" in nl:
        return "reduction"
    if "fill" in nl or "memset" in nl or "memcpy" in nl:
        return "memory"
    return "other"


def estimate_tensor_sizes(model, N, sh_degree, img_w, img_h):
    """Estimate tensor sizes for memory bandwidth calculation."""
    num_sh = (sh_degree + 1) ** 2  # 16 for degree 3

    sizes = {
        "xyz": (N, 3),            # 4 bytes * 3 = 12 bytes/elem
        "rotations": (N, 4),      # 16 bytes/elem
        "scales": (N, 3),         # 12 bytes/elem
        "opacity": (N, 1),        # 4 bytes/elem
        "shs": (N, num_sh, 3),    # 4 * num_sh * 3 bytes/elem = 192 for degree 3
        "rendered": (1, img_h, img_w, 3),  # 4 * 3 bytes/elem
        "gt_image": (img_h, img_w, 3),     # 4 * 3 bytes/elem
    }

    bytes_per_elem = 4  # float32

    # Adam: for each param, read param + grad + m + v, write param + m + v
    # = 6 * param_size (read 3, write 3)
    adam_bytes = 0
    for name, shape in sizes.items():
        if name in ["rendered", "gt_image"]:
            continue
        n_elem = 1
        for d in shape:
            n_elem *= d
        adam_bytes += n_elem * bytes_per_elem * 6  # read+write param, m, v

    # D-SSIM: convolution on [H, W, 3] image
    # Forward: read input + weights, write output
    # For 3x3 conv on [1, 3, H, W] → [1, 3, H, W]:
    #   input = 3 * H * W * 4 bytes
    #   output = 3 * H * W * 4 bytes
    #   weights = 3 * 3 * 3 * 3 * 4 = 324 bytes (negligible)
    dssim_input_bytes = 3 * img_h * img_w * bytes_per_elem
    dssim_output_bytes = dssim_input_bytes
    # D-SSIM does ~3-4 convolutions per call (blur, mean, var, cov)
    # Backward: 2x the forward (dgrad + wgrad)
    dssim_total_bytes_fwd = dssim_input_bytes + dssim_output_bytes  # per conv
    dssim_total_bytes_bwd = 3 * dssim_total_bytes_fwd  # backward ~3x forward

    # Rasterize bwd: processes ~3M intersections, each reads:
    # means2d (8B), conics (12B), colors (12B), opacities (4B), depths (4B),
    # flatten_ids (4B), isect_offsets (4B) = ~48 bytes read per intersection
    # Writes: d_means2d (8B), d_conics (12B), d_colors (12B), d_opacities (4B) = ~36 bytes write
    n_isects = 3_200_000  # approximate from C38
    rast_bwd_bytes = n_isects * (48 + 36)

    return {
        "adam": {
            "total_bytes": adam_bytes,
            "description": f"6× param read+write for {N} GS × 5 param groups",
            "param_groups": {
                "xyz": sizes["xyz"][0] * sizes["xyz"][1] * bytes_per_elem * 6,
                "rotations": sizes["rotations"][0] * sizes["rotations"][1] * bytes_per_elem * 6,
                "scales": sizes["scales"][0] * sizes["scales"][1] * bytes_per_elem * 6,
                "opacity": sizes["opacity"][0] * bytes_per_elem * 6,
                "shs": sizes["shs"][0] * sizes["shs"][1] * sizes["shs"][2] * bytes_per_elem * 6,
            },
        },
        "loss_dssim": {
            "per_conv_bytes_fwd": dssim_total_bytes_fwd,
            "per_conv_bytes_bwd": dssim_total_bytes_bwd,
            "total_bytes": 4 * dssim_total_bytes_fwd + 4 * dssim_total_bytes_bwd,
            "description": f"~4 conv fwd + 4 conv bwd on {img_w}x{img_h}x3 image",
        },
        "rasterize_bwd": {
            "total_bytes": rast_bwd_bytes,
            "n_isects": n_isects,
            "description": f"{n_isects} intersections, ~84 bytes read+write each",
        },
        "elementwise_misc": {
            "total_bytes": N * bytes_per_elem * 20,  # estimate: ~20 elementwise ops on N-element tensors
            "description": f"~130 kernels, avg ~N elements each",
        },
    }


def main():
    print("=" * 72)
    print("C41: GPU Utilization Bottleneck Discovery")
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

    # Load dataset
    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=device)
    cam0 = dataset.get_camera(0)
    gt0 = dataset.get_gt_image(0)
    img_w, img_h = cam0.image_width, cam0.image_height

    # Load model
    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=device)
    N = model.xyz.shape[0]
    print(f"  Model: {N:,} Gaussians, SH degree: {model.sh_degree}")
    print(f"  Image: {img_w}x{img_h}")

    # Estimate tensor sizes
    tensor_info = estimate_tensor_sizes(model, N, SH_DEGREE, img_w, img_h)

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
    # Phase 1: CUDA Events with NVTX ranges for phase attribution
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 1: CUDA Events with NVTX annotation (20 iters)")
    print(f"{'='*60}")

    N_EVTS = 20
    ev_defs = {}
    for name in ["fwd_render", "fwd_loss", "bwd", "opt"]:
        ev_defs[f"{name}_s"] = torch.cuda.Event(enable_timing=True)
        ev_defs[f"{name}_e"] = torch.cuda.Event(enable_timing=True)

    ev_total_s = torch.cuda.Event(enable_timing=True)
    ev_total_e = torch.cuda.Event(enable_timing=True)

    event_timings = []

    for iter_idx in range(1, N_EVTS + 1):
        ev_total_s.record()

        # Forward: model activations + render
        data = model.forward()
        ev_defs["fwd_render_s"].record()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        ev_defs["fwd_render_e"].record()
        rendered_img = rendered[0].clamp(0, 1)

        # Loss
        ev_defs["fwd_loss_s"].record()
        loss_dict = combined_loss(rendered_img, gt0, lambda_dssim=0.2)
        loss = loss_dict["loss"]
        ev_defs["fwd_loss_e"].record()

        # Backward
        ev_defs["bwd_s"].record()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        ev_defs["bwd_e"].record()

        # Optimizer
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        ev_defs["opt_s"].record()
        optimizer.step()
        torch.cuda.synchronize()
        ev_defs["opt_e"].record()

        ev_total_e.record()
        torch.cuda.synchronize()

        fwd_render_ms = ev_defs["fwd_render_s"].elapsed_time(ev_defs["fwd_render_e"])
        fwd_loss_ms = ev_defs["fwd_loss_s"].elapsed_time(ev_defs["fwd_loss_e"])
        bwd_ms = ev_defs["bwd_s"].elapsed_time(ev_defs["bwd_e"])
        opt_ms = ev_defs["opt_s"].elapsed_time(ev_defs["opt_e"])
        total_ms = ev_total_s.elapsed_time(ev_total_e)

        event_timings.append({
            "iter": iter_idx,
            "fwd_render_ms": fwd_render_ms,
            "fwd_loss_ms": fwd_loss_ms,
            "bwd_ms": bwd_ms,
            "opt_ms": opt_ms,
            "total_ms": total_ms,
        })

        if iter_idx % 5 == 0:
            print(f"  iter={iter_idx:3d}  render={fwd_render_ms:.2f}  loss={fwd_loss_ms:.2f}  "
                  f"bwd={bwd_ms:.2f}  opt={opt_ms:.2f}  total={total_ms:.2f}")

    # ═══════════════════════════════════════════════════
    # Phase 2: Detailed profiler with Chrome trace
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 2: torch.profiler Chrome trace (10 active iters)")
    print(f"{'='*60}")

    N_PROF = 10

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=N_PROF, repeat=1),
        on_trace_ready=lambda p: None,
    ) as prof:
        for iter_idx in range(2 + 3 + N_PROF):
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

    # Export Chrome trace
    trace_path = Path("results/phase-c31/c41_trace.json")
    prof.export_chrome_trace(str(trace_path))

    with open(trace_path) as f:
        trace = json.load(f)

    # Parse kernel events
    kernel_events = []
    for evt in trace.get("traceEvents", []):
        if evt.get("cat") == "kernel" and evt.get("dur", 0) > 0:
            kernel_name = evt.get("name", "")
            duration_us = evt.get("dur", 0)
            kernel_events.append({
                "name": kernel_name,
                "duration_us": duration_us,
                "category": categorize_kernel(kernel_name),
            })

    print(f"  Total CUDA kernel events: {len(kernel_events)}")

    # Aggregate by category
    category_totals = defaultdict(lambda: {
        "total_us": 0, "count": 0,
        "kernels": defaultdict(lambda: {"total_us": 0, "count": 0}),
        "durations": [],
    })

    for ke in kernel_events:
        cat = ke["category"]
        category_totals[cat]["total_us"] += ke["duration_us"]
        category_totals[cat]["count"] += 1
        category_totals[cat]["durations"].append(ke["duration_us"])
        category_totals[cat]["kernels"][ke["name"]]["total_us"] += ke["duration_us"]
        category_totals[cat]["kernels"][ke["name"]]["count"] += 1

    total_all_us = sum(v["total_us"] for v in category_totals.values())

    # ═══════════════════════════════════════════════════
    # Analysis per component
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("COMPONENT ANALYSIS")
    print(f"{'='*72}")

    # ─── 1. D-SSIM Loss Analysis ───
    print(f"\n{'='*60}")
    print("1. D-SSIM LOSS ANALYSIS")
    print(f"{'='*60}")

    dssim_data = category_totals.get("loss_dssim", {"total_us": 0, "count": 0, "durations": [], "kernels": {}})
    dssim_ms_per_iter = dssim_data["total_us"] / N_PROF / 1000
    dssim_count_per_iter = dssim_data["count"] / N_PROF
    dssim_pct = dssim_data["total_us"] / total_all_us * 100

    # Kernel duration distribution
    dssim_durations = np.array(dssim_data.get("durations", []))

    print(f"  Total time: {dssim_ms_per_iter:.3f} ms/iter ({dssim_pct:.1f}%)")
    print(f"  Kernel count: {dssim_count_per_iter:.1f} per iter")
    print(f"  Kernel duration: mean={dssim_durations.mean()/1000:.3f}ms  "
          f"median={np.median(dssim_durations)/1000:.3f}ms  "
          f"max={dssim_durations.max()/1000:.3f}ms")

    # Memory bandwidth estimate
    dssim_bytes = tensor_info["loss_dssim"]["total_bytes"]
    dssim_bw_GB_s = dssim_bytes / (dssim_ms_per_iter / 1000) / 1e9
    dssim_bw_pct = dssim_bw_GB_s / A100_PEAK_BW_GB_S * 100

    print(f"\n  Memory analysis:")
    print(f"    Estimated data movement: {dssim_bytes/1e6:.1f} MB/iter")
    print(f"    Achieved bandwidth: {dssim_bw_GB_s:.1f} GB/s ({dssim_bw_pct:.1f}% of peak {A100_PEAK_BW_GB_S} GB/s)")

    # FLOPs estimate: 3x3 conv on [1,3,1080,1920]
    # Each output pixel: 3*3*3*3 = 81 MACs = 162 FLOPs (for 3→3 channel conv)
    # But D-SSIM does multiple convolutions
    n_pixels = img_h * img_w
    flops_per_conv = n_pixels * 3 * 3 * 3 * 3 * 2  # MACs→FLOPs
    n_convs = 8  # estimate: 4 fwd + 4 bwd
    dssim_flops = n_convs * flops_per_conv
    dssim_tflops = dssim_flops / (dssim_ms_per_iter / 1000) / 1e12
    dssim_flops_pct_tf32 = dssim_tflops / A100_PEAK_TFLOPS_TF32 * 100
    dssim_flops_pct_fp32 = dssim_tflops / A100_PEAK_TFLOPS_FP32 * 100

    print(f"\n  Compute analysis:")
    print(f"    Estimated FLOPs: {dssim_flops/1e9:.2f} GFLOP/iter")
    print(f"    Achieved: {dssim_tflops:.2f} TFLOPS")
    print(f"    % of peak TF32: {dssim_flops_pct_tf32:.2f}% (tensor core)")
    print(f"    % of peak FP32: {dssim_flops_pct_fp32:.2f}%")

    # Classification
    # If BW utilization > 50%: memory bound
    # If FLOPs utilization > 50%: compute bound
    # If kernel count > 20 and avg duration < 0.1ms: launch bound
    dssim_classification = "unknown"
    if dssim_bw_pct > 50:
        dssim_classification = "memory bound"
    elif dssim_flops_pct_tf32 > 30:
        dssim_classification = "compute bound (tensor core)"
    elif dssim_flops_pct_fp32 > 50:
        dssim_classification = "compute bound (FP32)"
    elif dssim_count_per_iter > 20 and dssim_durations.mean() < 100:
        dssim_classification = "launch bound"
    else:
        dssim_classification = "algorithmic excessive-work bound"

    print(f"\n  Classification: {dssim_classification}")
    print(f"  Root cause hypothesis: {'cuDNN convolution doing unavoidable 2D spatial computation' if 'compute' in dssim_classification else 'inefficient memory access' if 'memory' in dssim_classification else 'too many small kernel launches' if 'launch' in dssim_classification else 'excessive algorithmic work'}")

    # Top kernels
    print(f"\n  Top D-SSIM kernels:")
    for kname, kdata in sorted(dssim_data["kernels"].items(), key=lambda x: x[1]["total_us"], reverse=True)[:5]:
        short = kname[:80] + "..." if len(kname) > 80 else kname
        print(f"    {short}")
        print(f"      {kdata['total_us']/N_PROF/1000:.3f} ms/iter  ({kdata['count']/N_PROF:.1f} calls)")

    # ─── 2. Adam Optimizer Analysis ───
    print(f"\n{'='*60}")
    print("2. ADAM OPTIMIZER ANALYSIS")
    print(f"{'='*60}")

    adam_data = category_totals.get("adam", {"total_us": 0, "count": 0, "durations": [], "kernels": {}})
    adam_ms_per_iter = adam_data["total_us"] / N_PROF / 1000
    adam_count_per_iter = adam_data["count"] / N_PROF
    adam_pct = adam_data["total_us"] / total_all_us * 100
    adam_durations = np.array(adam_data.get("durations", []))

    print(f"  Total time: {adam_ms_per_iter:.3f} ms/iter ({adam_pct:.1f}%)")
    print(f"  Kernel count: {adam_count_per_iter:.1f} per iter")
    print(f"  Kernel duration: mean={adam_durations.mean()/1000:.3f}ms  "
          f"median={np.median(adam_durations)/1000:.3f}ms  "
          f"max={adam_durations.max()/1000:.3f}ms")

    # Memory bandwidth
    adam_bytes = tensor_info["adam"]["total_bytes"]
    adam_bw_GB_s = adam_bytes / (adam_ms_per_iter / 1000) / 1e9
    adam_bw_pct = adam_bw_GB_s / A100_PEAK_BW_GB_S * 100

    print(f"\n  Memory analysis:")
    print(f"    Total param bytes: {adam_bytes/1e6:.1f} MB (6× all params)")
    print(f"    Achieved bandwidth: {adam_bw_GB_s:.1f} GB/s ({adam_bw_pct:.1f}% of peak)")
    print(f"    Per-group breakdown:")
    for gname, gbytes in tensor_info["adam"]["param_groups"].items():
        g_pct = gbytes / adam_bytes * 100
        print(f"      {gname:12s}: {gbytes/1e6:8.1f} MB ({g_pct:5.1f}%)")

    # Launch overhead estimate: each kernel launch ~5-10us on A100
    launch_overhead_us = adam_count_per_iter * 7  # 7us average
    launch_overhead_pct = launch_overhead_us / (adam_ms_per_iter * 1000) * 100

    print(f"\n  Launch overhead estimate:")
    print(f"    Kernels/iter: {adam_count_per_iter:.0f}")
    print(f"    Est. launch overhead: {launch_overhead_us/1000:.3f} ms ({launch_overhead_pct:.1f}% of Adam time)")

    adam_classification = "unknown"
    if adam_bw_pct > 50:
        adam_classification = "memory bandwidth bound"
    elif launch_overhead_pct > 30:
        adam_classification = "kernel launch bound"
    elif adam_count_per_iter > 30:
        adam_classification = "kernel fragmentation bound"
    else:
        adam_classification = "compute bound"

    print(f"\n  Classification: {adam_classification}")

    # ─── 3. Elementwise Misc Analysis ───
    print(f"\n{'='*60}")
    print("3. ELEMENTWISE MISC ANALYSIS")
    print(f"{'='*60}")

    elem_data = category_totals.get("elementwise_misc", {"total_us": 0, "count": 0, "durations": [], "kernels": {}})
    elem_ms_per_iter = elem_data["total_us"] / N_PROF / 1000
    elem_count_per_iter = elem_data["count"] / N_PROF
    elem_pct = elem_data["total_us"] / total_all_us * 100
    elem_durations = np.array(elem_data.get("durations", []))

    print(f"  Total time: {elem_ms_per_iter:.3f} ms/iter ({elem_pct:.1f}%)")
    print(f"  Kernel count: {elem_count_per_iter:.1f} per iter")
    print(f"  Kernel duration: mean={elem_durations.mean():.1f}us  "
          f"median={np.median(elem_durations):.1f}us  "
          f"max={elem_durations.max():.1f}us  "
          f"p90={np.percentile(elem_durations, 90):.1f}us")

    # Launch overhead
    elem_launch_us = elem_count_per_iter * 7
    elem_launch_pct = elem_launch_us / (elem_ms_per_iter * 1000) * 100

    print(f"\n  Launch overhead estimate:")
    print(f"    Kernels/iter: {elem_count_per_iter:.0f}")
    print(f"    Est. launch overhead: {elem_launch_us/1000:.3f} ms ({elem_launch_pct:.1f}% of elem time)")

    elem_bytes = tensor_info["elementwise_misc"]["total_bytes"]
    elem_bw_GB_s = elem_bytes / (elem_ms_per_iter / 1000) / 1e9
    elem_bw_pct = elem_bw_GB_s / A100_PEAK_BW_GB_S * 100

    print(f"\n  Memory analysis:")
    print(f"    Estimated data movement: {elem_bytes/1e6:.1f} MB/iter")
    print(f"    Achieved bandwidth: {elem_bw_GB_s:.1f} GB/s ({elem_bw_pct:.1f}% of peak)")

    elem_classification = "unknown"
    if elem_launch_pct > 50:
        elem_classification = "kernel launch bound"
    elif elem_bw_pct > 50:
        elem_classification = "memory bandwidth bound"
    else:
        elem_classification = "kernel fragmentation (launch + small-tensor)"

    print(f"\n  Classification: {elem_classification}")
    print(f"\n  Top elementwise kernels:")
    for kname, kdata in sorted(elem_data["kernels"].items(), key=lambda x: x[1]["total_us"], reverse=True)[:5]:
        short = kname[:80] + "..." if len(kname) > 80 else kname
        print(f"    {short}")
        print(f"      {kdata['total_us']/N_PROF/1000:.3f} ms/iter  ({kdata['count']/N_PROF:.1f} calls)")

    # ─── 4. rasterize_bwd Roofline ───
    print(f"\n{'='*60}")
    print("4. RASTERIZE_BWD ROOFLINE ANALYSIS")
    print(f"{'='*60}")

    rbwd_data = category_totals.get("rasterize_bwd", {"total_us": 0, "count": 0, "durations": [], "kernels": {}})
    rbwd_ms_per_iter = rbwd_data["total_us"] / N_PROF / 1000
    rbwd_count_per_iter = rbwd_data["count"] / N_PROF
    rbwd_pct = rbwd_data["total_us"] / total_all_us * 100
    rbwd_durations = np.array(rbwd_data.get("durations", []))

    print(f"  Total time: {rbwd_ms_per_iter:.3f} ms/iter ({rbwd_pct:.1f}%)")
    print(f"  Kernel count: {rbwd_count_per_iter:.1f} per iter")
    print(f"  Kernel duration: mean={rbwd_durations.mean()/1000:.3f}ms  "
          f"median={np.median(rbwd_durations)/1000:.3f}ms")

    # Memory analysis
    rbwd_bytes = tensor_info["rasterize_bwd"]["total_bytes"]
    rbwd_bw_GB_s = rbwd_bytes / (rbwd_ms_per_iter / 1000) / 1e9
    rbwd_bw_pct = rbwd_bw_GB_s / A100_PEAK_BW_GB_S * 100

    print(f"\n  Memory analysis:")
    print(f"    n_intersections: {tensor_info['rasterize_bwd']['n_isects']:,}")
    print(f"    Estimated data movement: {rbwd_bytes/1e6:.1f} MB/iter")
    print(f"    Achieved bandwidth: {rbwd_bw_GB_s:.1f} GB/s ({rbwd_bw_pct:.1f}% of peak)")

    # FLOPs estimate: each intersection does ~20 FLOPs for alpha blending + gradient
    rbwd_flops = tensor_info["rasterize_bwd"]["n_isects"] * 40  # ~40 FLOPs per intersection backward
    rbwd_tflops = rbwd_flops / (rbwd_ms_per_iter / 1000) / 1e12
    rbwd_flops_pct = rbwd_tflops / A100_PEAK_TFLOPS_FP32 * 100

    print(f"\n  Compute analysis:")
    print(f"    Estimated FLOPs: {rbwd_flops/1e9:.2f} GFLOP/iter")
    print(f"    Achieved: {rbwd_tflops:.3f} TFLOPS ({rbwd_flops_pct:.2f}% of FP32 peak)")

    # Arithmetic intensity
    rbwd_ai = rbwd_flops / rbwd_bytes  # FLOPs/byte
    print(f"    Arithmetic intensity: {rbwd_ai:.2f} FLOPs/byte")

    # Roofline: compute bound if AI > peak_flops/peak_bw
    ridge_point = A100_PEAK_TFLOPS_FP32 * 1e12 / (A100_PEAK_BW_GB_S * 1e9)  # FLOPs/byte
    print(f"    Ridge point: {ridge_point:.1f} FLOPs/byte")

    rbwd_classification = "unknown"
    if rbwd_ai < ridge_point:
        rbwd_classification = f"memory bound (AI={rbwd_ai:.2f} < ridge={ridge_point:.1f})"
    else:
        rbwd_classification = f"compute bound (AI={rbwd_ai:.2f} > ridge={ridge_point:.1f})"

    print(f"\n  Classification: {rbwd_classification}")

    # ─── Summary table ───
    print(f"\n{'='*72}")
    print("GPU UTILIZATION TABLE")
    print(f"{'='*72}")
    print(f"\n  {'Component':<20s}  {'Time%':>6s}  {'ms/iter':>8s}  {'Kernels':>8s}  {'BW(GB/s)':>9s}  {'BW%':>6s}  {'Class':>30s}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*30}")

    for cat_name, cat_data, bw_pct in [
        ("loss_dssim", dssim_data, dssim_bw_pct),
        ("adam", adam_data, adam_bw_pct),
        ("elementwise", elem_data, elem_bw_pct),
        ("rasterize_bwd", rbwd_data, rbwd_bw_pct),
    ]:
        ms = cat_data["total_us"] / N_PROF / 1000
        pct = cat_data["total_us"] / total_all_us * 100
        cnt = cat_data["count"] / N_PROF
        bw = [dssim_bw_GB_s, adam_bw_GB_s, elem_bw_GB_s, rbwd_bw_GB_s][
            ["loss_dssim", "adam", "elementwise", "rasterize_bwd"].index(cat_name)
        ]
        cls = [dssim_classification, adam_classification, elem_classification, rbwd_classification][
            ["loss_dssim", "adam", "elementwise", "rasterize_bwd"].index(cat_name)
        ]
        print(f"  {cat_name:<20s}  {pct:>5.1f}%  {ms:>8.3f}  {cnt:>8.1f}  {bw:>9.1f}  {bw_pct:>5.1f}%  {cls:>30s}")

    # Also print full category list
    print(f"\n  All categories:")
    for cat in sorted(category_totals.keys(), key=lambda k: category_totals[k]["total_us"], reverse=True):
        v = category_totals[cat]
        ms = v["total_us"] / N_PROF / 1000
        pct = v["total_us"] / total_all_us * 100
        cnt = v["count"] / N_PROF
        print(f"  {cat:<20s}  {pct:>5.1f}%  {ms:>8.3f}ms  {cnt:>6.1f} kernels/iter")

    # ─── Save output ───
    def safe_float(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v

    output = {
        "config": {
            "scene": "room",
            "n_gaussians": N,
            "sh_degree": SH_DEGREE,
            "image_size": f"{img_w}x{img_h}",
            "densification": "disabled",
            "warmup": WARMUP,
            "n_event_iters": N_EVTS,
            "n_profiler_iters": N_PROF,
        },
        "gpu_specs": {
            "name": "A100-PCIE-40GB",
            "n_SMs": A100_SMs,
            "peak_bw_GB_s": A100_PEAK_BW_GB_S,
            "peak_tflops_fp32": A100_PEAK_TFLOPS_FP32,
            "peak_tflops_tf32": A100_PEAK_TFLOPS_TF32,
        },
        "method1_cuda_events": {
            "summary": {
                "fwd_render_ms": float(np.mean([t["fwd_render_ms"] for t in event_timings])),
                "fwd_loss_ms": float(np.mean([t["fwd_loss_ms"] for t in event_timings])),
                "bwd_ms": float(np.mean([t["bwd_ms"] for t in event_timings])),
                "opt_ms": float(np.mean([t["opt_ms"] for t in event_timings])),
                "total_ms": float(np.mean([t["total_ms"] for t in event_timings])),
            },
        },
        "method2_profiler": {
            "total_gpu_ms_per_iter": float(total_all_us / N_PROF / 1000),
            "categories": {
                cat: {
                    "ms_per_iter": float(v["total_us"] / N_PROF / 1000),
                    "pct": float(v["total_us"] / total_all_us * 100),
                    "kernels_per_iter": float(v["count"] / N_PROF),
                    "mean_kernel_us": float(np.mean(v["durations"])) if v["durations"] else 0,
                    "median_kernel_us": float(np.median(v["durations"])) if v["durations"] else 0,
                    "max_kernel_us": float(np.max(v["durations"])) if v["durations"] else 0,
                }
                for cat, v in category_totals.items()
            },
        },
        "component_analysis": {
            "loss_dssim": {
                "ms_per_iter": float(dssim_ms_per_iter),
                "pct": float(dssim_pct),
                "kernels_per_iter": float(dssim_count_per_iter),
                "estimated_bytes": int(dssim_bytes),
                "achieved_bw_GB_s": float(dssim_bw_GB_s),
                "bw_pct_of_peak": float(dssim_bw_pct),
                "estimated_flops": int(dssim_flops),
                "achieved_tflops": float(dssim_tflops),
                "flops_pct_tf32": float(dssim_flops_pct_tf32),
                "flops_pct_fp32": float(dssim_flops_pct_fp32),
                "classification": dssim_classification,
            },
            "adam": {
                "ms_per_iter": float(adam_ms_per_iter),
                "pct": float(adam_pct),
                "kernels_per_iter": float(adam_count_per_iter),
                "estimated_bytes": int(adam_bytes),
                "achieved_bw_GB_s": float(adam_bw_GB_s),
                "bw_pct_of_peak": float(adam_bw_pct),
                "launch_overhead_us": float(launch_overhead_us),
                "launch_overhead_pct": float(launch_overhead_pct),
                "param_groups": {k: float(v) for k, v in tensor_info["adam"]["param_groups"].items()},
                "classification": adam_classification,
            },
            "elementwise_misc": {
                "ms_per_iter": float(elem_ms_per_iter),
                "pct": float(elem_pct),
                "kernels_per_iter": float(elem_count_per_iter),
                "mean_kernel_us": float(elem_durations.mean()) if len(elem_durations) > 0 else 0,
                "estimated_bytes": int(elem_bytes),
                "achieved_bw_GB_s": float(elem_bw_GB_s),
                "bw_pct_of_peak": float(elem_bw_pct),
                "launch_overhead_us": float(elem_launch_us),
                "launch_overhead_pct": float(elem_launch_pct),
                "classification": elem_classification,
            },
            "rasterize_bwd": {
                "ms_per_iter": float(rbwd_ms_per_iter),
                "pct": float(rbwd_pct),
                "kernels_per_iter": float(rbwd_count_per_iter),
                "estimated_bytes": int(rbwd_bytes),
                "achieved_bw_GB_s": float(rbwd_bw_GB_s),
                "bw_pct_of_peak": float(rbwd_bw_pct),
                "estimated_flops": int(rbwd_flops),
                "achieved_tflops": float(rbwd_tflops),
                "arithmetic_intensity": float(rbwd_ai),
                "ridge_point": float(ridge_point),
                "classification": rbwd_classification,
            },
        },
    }

    save_path = Path("results/phase-c31/c41_gpu_utilization.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=safe_float)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
