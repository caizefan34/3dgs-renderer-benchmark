#!/usr/bin/env python3
"""
C40 A100 Revalidation: Training Kernel Bottleneck Profiling.

A100 version of C40 — uses A100-trained checkpoint.
Measures the same breakdown as the original C40 but on A100-PCIE-40GB.
"""
import json, math, os, time, sys
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


def categorize_kernel(name):
    nl = name.lower()
    if "fully_fused_projection" in nl and "bwd" not in nl:
        return "projection_fwd"
    if "isect_tiles" in nl or "isect_offset" in nl:
        return "intersection"
    if "rasterize_to_pixels" in nl and "bwd" not in nl:
        return "rasterize_fwd"
    if "spherical_harmonics" in nl and "bwd" not in nl:
        return "projection_fwd"
    if "rasterize_to_pixels" in nl and "bwd" in nl:
        return "rasterize_bwd"
    if "fully_fused_projection" in nl and "bwd" in nl:
        return "projection_bwd"
    if "spherical_harmonics" in nl and "bwd" in nl:
        return "projection_bwd"
    if "radixsort" in nl or "radix_sort" in nl or "segmentedsort" in nl:
        return "sorting"
    if "cub" in nl and "sort" in nl:
        return "sorting"
    if "multi_tensor_apply" in nl or "adam" in nl:
        return "adam"
    if "cutlass" in nl or "cudnn" in nl:
        return "loss_dssim"
    if "tensorTransform" in nl:
        return "loss_dssim"
    if "dgrad" in nl or "wgrad" in nl or "fgrad" in nl:
        return "loss_dssim"
    if "convolution" in nl or "conv_" in nl:
        return "loss_dssim"
    if "memcpy" in nl or "memset" in nl:
        return "memory"
    if "fill" in nl and "elementwise" in nl:
        return "memory"
    if "elementwise" in nl or "pointwise" in nl or "binary" in nl:
        return "elementwise_misc"
    if "reduce" in nl or "reduction" in nl:
        return "reduction"
    return "other"


def main():
    print("=" * 72)
    print("C40 A100 Revalidation: Training Kernel Bottleneck Profiling")
    print("=" * 72)

    # Record hardware fingerprint
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {gpu_name}")
    print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    print(f"  VRAM: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"  SMs: {gpu_props.multi_processor_count}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    import gsplat
    print(f"  gsplat: {gsplat.__version__}")

    device = "cuda"
    torch.manual_seed(42)

    TILE_SIZE = 16
    PACKED = True
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
    print(f"  Camera 0: {img_w}x{img_h}")

    # Load A100-trained checkpoint
    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "a100_30k_room_t16_16" / "a100_30k_room_t16_16_iter5000.pt"
    if not ckpt_path.exists():
        # Try storage pool
        alt = Path("/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7/a100_30k_room_t16_16/a100_30k_room_t16_16_iter5000.pt")
        if alt.exists():
            ckpt_path = alt
        else:
            print(f"  ERROR: Checkpoint not found at {ckpt_path} or {alt}")
            sys.exit(1)

    print(f"\n  Loading A100 checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=device)
    N = model.xyz.shape[0]
    print(f"  Model: {N:,} Gaussians, SH degree: {model.sh_degree}")
    print(f"  Checkpoint iteration: {ckpt.get('iteration', 'unknown')}")
    print(f"  Checkpoint PSNR: {ckpt.get('metrics', {}).get('psnr', 'unknown')}")

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * SPATIAL_LR_SCALE, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    WARMUP = 30
    N_PROFILE = 20

    # ── Warmup ──
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
    # METHOD 1: CUDA events
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Method 1: CUDA Events — {N_PROFILE} iterations")
    print(f"{'='*60}")

    evs = {k: torch.cuda.Event(enable_timing=True) for k in
           ["fwd_s", "fwd_e", "bwd_s", "bwd_e", "opt_s", "opt_e", "loss_s", "loss_e", "total_s", "total_e"]}

    event_timings = []
    for iter_idx in range(1, N_PROFILE + 1):
        evs["total_s"].record()
        data = model.forward()
        evs["fwd_s"].record()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        evs["fwd_e"].record()
        rendered_img = rendered[0].clamp(0, 1)

        evs["loss_s"].record()
        loss_dict = combined_loss(rendered_img, gt0, lambda_dssim=0.2)
        loss = loss_dict["loss"]
        evs["loss_e"].record()

        evs["bwd_s"].record()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        evs["bwd_e"].record()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        evs["opt_s"].record()
        optimizer.step()
        torch.cuda.synchronize()
        evs["opt_e"].record()

        evs["total_e"].record()
        torch.cuda.synchronize()

        fwd_ms = evs["fwd_s"].elapsed_time(evs["fwd_e"])
        loss_ms = evs["loss_s"].elapsed_time(evs["loss_e"])
        bwd_ms = evs["bwd_s"].elapsed_time(evs["bwd_e"])
        opt_ms = evs["opt_s"].elapsed_time(evs["opt_e"])
        total_ms = evs["total_s"].elapsed_time(evs["total_e"])

        event_timings.append({"iter": iter_idx, "fwd_ms": fwd_ms, "loss_ms": loss_ms,
                              "bwd_ms": bwd_ms, "opt_ms": opt_ms, "total_ms": total_ms})
        if iter_idx % 5 == 0:
            print(f"  iter={iter_idx:3d}  fwd={fwd_ms:.2f}ms  loss={loss_ms:.2f}ms  "
                  f"bwd={bwd_ms:.2f}ms  opt={opt_ms:.2f}ms  total={total_ms:.2f}ms")

    # Summary
    arr = {k: np.array([t[k] for t in event_timings]) for k in ["fwd_ms", "loss_ms", "bwd_ms", "opt_ms", "total_ms"]}
    print(f"\n  Summary (mean):")
    total_mean = arr["total_ms"].mean()
    for k in ["fwd_ms", "loss_ms", "bwd_ms", "opt_ms", "total_ms"]:
        pct = arr[k].mean() / total_mean * 100 if k != "total_ms" else 100
        print(f"    {k:12s}: {arr[k].mean():.2f} +/- {arr[k].std():.2f} ms  ({pct:.1f}%)")

    # ═══════════════════════════════════════════════════
    # METHOD 2: Block timing (wall clock, no CUDA events)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Method 2: Block Timing (wall clock, 150 iters)")
    print(f"{'='*60}")

    # Warmup for block timing
    for i in range(10):
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
    torch.cuda.synchronize()

    block_times = []
    for i in range(150):
        t0 = time.perf_counter()
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
        torch.cuda.synchronize()
        block_times.append((time.perf_counter() - t0) * 1000)

    block_arr = np.array(block_times)
    print(f"  Block T_iter: {block_arr.mean():.2f} +/- {block_arr.std():.2f} ms")
    print(f"  Min: {block_arr.min():.2f}  Max: {block_arr.max():.2f}")

    # ═══════════════════════════════════════════════════
    # METHOD 3: torch.profiler kernel breakdown
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Method 3: torch.profiler (10 profiled iterations)")
    print(f"{'='*60}")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for i in range(10):
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

    # Parse profiler events
    events = prof.events()
    kernel_times = defaultdict(lambda: {"total_us": 0, "count": 0})
    for ev in events:
        if ev.device_type == torch.profiler.DeviceType.CUDA:
            name = ev.name
            # PyTorch 2.7: use self_device_time_total (microseconds)
            dur_us = ev.self_device_time_total if hasattr(ev, 'self_device_time_total') else 0
            if dur_us and dur_us > 0:
                cat = categorize_kernel(name)
                kernel_times[cat]["total_us"] += dur_us
                kernel_times[cat]["count"] += 1

    total_us = sum(v["total_us"] for v in kernel_times.values())
    print(f"\n  Total GPU kernel time: {total_us / 1000:.2f} ms/iter (over 10 iters)")
    print(f"\n  {'Category':25s}  {'Time/iter (ms)':>14s}  {'%':>6s}  {'Kernels/iter':>12s}")
    print("  " + "-" * 65)
    for cat in sorted(kernel_times.keys(), key=lambda c: kernel_times[c]["total_us"], reverse=True):
        t = kernel_times[cat]
        ms_per_iter = t["total_us"] / 1000 / 10
        pct = t["total_us"] / total_us * 100
        k_per_iter = t["count"] / 10
        print(f"  {cat:25s}  {ms_per_iter:>14.3f}  {pct:>5.1f}%  {k_per_iter:>12.1f}")

    # ── Save ──
    output = {
        "experiment": "C40 A100 Revalidation",
        "hardware": {
            "gpu": gpu_name,
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "vram_gb": gpu_props.total_memory / 1e9,
            "sms": gpu_props.multi_processor_count,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "iteration": ckpt.get("iteration"),
            "n_gaussians": N,
            "sh_degree": model.sh_degree,
            "psnr": ckpt.get("metrics", {}).get("psnr"),
        },
        "method1_cuda_events": {
            "per_iter": event_timings,
            "summary": {k: {"mean": float(arr[k].mean()), "std": float(arr[k].std()),
                            "p90": float(np.percentile(arr[k], 90))}
                        for k in arr},
        },
        "method2_block_timing": {
            "mean_ms": float(block_arr.mean()),
            "std_ms": float(block_arr.std()),
            "min_ms": float(block_arr.min()),
            "max_ms": float(block_arr.max()),
        },
        "method3_profiler": {
            "total_gpu_ms_per_iter": total_us / 1000 / 10,
            "categories": {cat: {"ms_per_iter": t["total_us"] / 1000 / 10,
                                 "pct": t["total_us"] / total_us * 100,
                                 "kernels_per_iter": t["count"] / 10}
                           for cat, t in kernel_times.items()},
        },
    }

    save_path = repo_root / "results" / "a100" / "validation-c40-c42" / "c40_baseline_a100.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")

    # ── Key question ──
    dssim_pct = kernel_times.get("loss_dssim", {}).get("total_us", 0) / total_us * 100
    print(f"\n{'='*72}")
    print(f"KEY QUESTION: Is D-SSIM still a significant bottleneck on A100?")
    print(f"  D-SSIM share of GPU kernel time: {dssim_pct:.1f}%")
    print(f"  (RTX 5070 measured: 40.7%)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
