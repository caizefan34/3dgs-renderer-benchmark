#!/usr/bin/env python3
"""
C42 Phase 1-C: Separable Conv + torch.compile Interaction Validation.

Three variants:
  V0: Original 2D conv, no compile
  V1: Separable conv, no compile
  V2: Separable conv, torch.compile ON

Measurements:
  1. D-SSIM forward latency (isolated, pre-rendered image)
  2. D-SSIM backward latency (isolated)
  3. Full forward+backward latency (isolated)
  4. CUDA kernel count (profiler)
  5. GPU memory allocation
  6. End-to-end training iteration time (render + loss + bwd + opt)
"""
import json, math, sys, time, gc
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

# ── Config ──
DEVICE = "cuda"
WINDOW = 11
SIGMA = 1.5
N_WARMUP = 30
N_MEASURE = 50
N_PROF = 10
TILE_SIZE = 16
PACKED = True
SH_DEGREE = 3
EPS2D = 0.1
RADIUS_CLIP = 0.0
SPATIAL_LR_SCALE = 46.64
LAMBDA_DSSIM = 0.2


# ═══════════════════════════════════════════════════════════════
# Loss function definitions
# ═══════════════════════════════════════════════════════════════

def d_ssim_2d(pred, target, window_size=WINDOW, sigma=SIGMA, data_range=1.0):
    """Original 2D convolution D-SSIM — matches loss.py exactly."""
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


def d_ssim_sep(pred, target, window_size=WINDOW, sigma=SIGMA, data_range=1.0):
    """Separable convolution D-SSIM — 1D horizontal + 1D vertical."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kh = kernel_1d.view(1, 1, 1, window_size).expand(3, 1, 1, window_size).contiguous()
    kv = kernel_1d.view(1, 1, window_size, 1).expand(3, 1, window_size, 1).contiguous()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    def blur(x):
        x_h = F.conv2d(x, kh, padding=(0, window_size // 2), groups=3)
        return F.conv2d(x_h, kv, padding=(window_size // 2, 0), groups=3)
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


def combined_2d(pred, target, lambda_dssim=LAMBDA_DSSIM):
    l1 = F.l1_loss(pred, target)
    dsim = d_ssim_2d(pred, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dsim


def combined_sep(pred, target, lambda_dssim=LAMBDA_DSSIM):
    l1 = F.l1_loss(pred, target)
    dsim = d_ssim_sep(pred, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dsim


# V2: compiled separable — use aot_eager (Triton/inductor unavailable on Windows)
# Also test cudagraphs backend as V2b (eliminates kernel launch overhead)
print("Compiling V2 (torch.compile, backend=aot_eager)...")
try:
    combined_sep_compiled = torch.compile(combined_sep, backend="aot_eager")
    V2_AVAILABLE = True
except Exception as e:
    print(f"  aot_eager failed: {e}")
    combined_sep_compiled = combined_sep  # fallback
    V2_AVAILABLE = False

print("Compiling V2b (torch.compile, backend=cudagraphs)...")
try:
    combined_sep_cg = torch.compile(combined_sep, backend="cudagraphs")
    V2B_AVAILABLE = True
except Exception as e:
    print(f"  cudagraphs failed: {e}")
    combined_sep_cg = combined_sep  # fallback
    V2B_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# Timing helpers
# ═══════════════════════════════════════════════════════════════

def cuda_time(fn, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_measure):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def time_backward_only(loss_fn, pred_base, target, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    """Time only the backward pass (forward done untimed beforehand)."""
    for _ in range(n_warmup):
        pred = pred_base.clone().requires_grad_(True)
        loss = loss_fn(pred, target)
        loss.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_measure):
        pred = pred_base.clone().requires_grad_(True)
        loss = loss_fn(pred, target)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        loss.backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def count_kernels(fn, n_iters=N_PROF):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=n_iters, repeat=1),
    ) as prof:
        for _ in range(1 + 2 + n_iters):
            fn()
            prof.step()
    trace_path = Path("results/phase-c42/c42_compile_trace_tmp.json")
    prof.export_chrome_trace(str(trace_path))
    with open(trace_path) as f:
        trace = json.load(f)
    kernels = defaultdict(lambda: {"total_us": 0, "count": 0})
    for evt in trace.get("traceEvents", []):
        if evt.get("cat") == "kernel" and evt.get("dur", 0) > 0:
            kernels[evt["name"]]["total_us"] += evt["dur"]
            kernels[evt["name"]]["count"] += 1
    return kernels, n_iters, trace_path


def categorize(name):
    nl = name.lower()
    if "dgrad" in nl:
        return "dgrad (bwd conv)"
    elif "conv2d" in nl or "depthwise" in nl:
        return "conv2d (fwd conv)"
    elif "triton" in nl:
        return "triton (compiled)"
    elif "elementwise" in nl or "vectorized" in nl or "pow" in nl or "neg" in nl or "pointwise" in nl:
        return "elementwise"
    elif "reduce" in nl:
        return "reduction"
    else:
        return "other"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("C42 Phase 1-C: Separable Conv + torch.compile Validation")
    print("=" * 72)

    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parent.parent.parent

    # ── 1. Load model and dataset ──
    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    cam0 = dataset.get_camera(0)
    gt0 = dataset.get_gt_image(0)
    img_w, img_h = cam0.image_width, cam0.image_height

    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=DEVICE)
    N = model.xyz.shape[0]
    print(f"  Model: {N:,} Gaussians, Image: {img_w}x{img_h}")

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * SPATIAL_LR_SCALE, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    variants = [
        ("V0", combined_2d),
        ("V1", combined_sep),
    ]
    if V2_AVAILABLE:
        variants.append(("V2", combined_sep_compiled))
    if V2B_AVAILABLE:
        variants.append(("V2b", combined_sep_cg))

    # ── 2. Pre-render image for isolated tests ──
    print("\n  [Pre-rendering image...]")
    with torch.no_grad():
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        pred_base = rendered[0].clamp(0, 1).detach()
    print(f"  Pre-rendered: shape={pred_base.shape}, range=[{pred_base.min():.3f}, {pred_base.max():.3f}]")

    # ── 3. Correctness ──
    print(f"\n{'='*60}")
    print("Section 3: Correctness")
    print(f"{'='*60}")

    correctness = {}
    for vname, vfn in variants:
        pred_v = pred_base.clone().requires_grad_(True)
        loss_v = vfn(pred_v, gt0)
        loss_v.backward()
        grad_v = pred_v.grad.clone()
        correctness[vname] = {"loss": loss_v.item(), "grad": grad_v}

    # Compare V0 vs all others
    for cmp in [k for k in correctness.keys() if k != "V0"]:
        loss_diff = abs(correctness["V0"]["loss"] - correctness[cmp]["loss"])
        grad_diff = (correctness["V0"]["grad"] - correctness[cmp]["grad"]).abs().max().item()
        print(f"  V0 vs {cmp}: loss_diff={loss_diff:.2e}  grad_diff={grad_diff:.2e}")

    v0v2_loss_diff = abs(correctness.get("V2", correctness.get("V2b", correctness["V0"]))["loss"] - correctness["V0"]["loss"]) if (V2_AVAILABLE or V2B_AVAILABLE) else float('inf')
    v0v2_grad_diff = (correctness.get("V2", correctness.get("V2b", correctness["V0"]))["grad"] - correctness["V0"]["grad"]).abs().max().item() if (V2_AVAILABLE or V2B_AVAILABLE) else float('inf')
    v2_name = "V2" if V2_AVAILABLE else ("V2b" if V2B_AVAILABLE else "N/A")
    print(f"\n  Gate: loss_diff < 1e-5, grad_diff < 1e-4")
    print(f"  V0 vs V2: {'PASS' if v0v2_loss_diff < 1e-5 and v0v2_grad_diff < 1e-4 else 'FAIL'}")

    # ── 4. V2 Compilation timing ──
    print(f"\n{'='*60}")
    print("Section 4: V2 Compilation Timing")
    print(f"{'='*60}")
    compile_time = 0.0
    if V2_AVAILABLE:
        t0 = time.perf_counter()
        for _ in range(5):
            pred = pred_base.clone().requires_grad_(True)
            loss = combined_sep_compiled(pred, gt0)
            loss.backward()
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t0
        print(f"  V2 (aot_eager) compilation + first 5 calls: {compile_time:.2f}s")
    else:
        print("  V2 (aot_eager) not available")
    if V2B_AVAILABLE:
        t0 = time.perf_counter()
        for _ in range(5):
            pred = pred_base.clone().requires_grad_(True)
            loss = combined_sep_cg(pred, gt0)
            loss.backward()
        torch.cuda.synchronize()
        compile_time_cg = time.perf_counter() - t0
        print(f"  V2b (cudagraphs) compilation + first 5 calls: {compile_time_cg:.2f}s")
    else:
        print("  V2b (cudagraphs) not available")

    # ── 5. Isolated timing (forward, backward, full fwd+bwd) ──
    print(f"\n{'='*60}")
    print(f"Section 5: Isolated Timing ({N_MEASURE} measurements each)")
    print(f"{'='*60}")

    isolated_results = {}
    for vname, vfn in variants:
        print(f"\n  --- {vname} ---")

        # Forward only
        mean_fwd, std_fwd, min_fwd, max_fwd = cuda_time(lambda: vfn(pred_base.clone(), gt0))
        print(f"  Forward:    {mean_fwd:.3f} +/- {std_fwd:.3f} ms")

        # Backward only
        mean_bwd, std_bwd = time_backward_only(vfn, pred_base, gt0)
        print(f"  Backward:   {mean_bwd:.3f} +/- {std_bwd:.3f} ms")

        # Full fwd+bwd
        def full_fn():
            pred = pred_base.clone().requires_grad_(True)
            loss = vfn(pred, gt0)
            loss.backward()
        mean_full, std_full, _, _ = cuda_time(full_fn)
        print(f"  Fwd+Bwd:    {mean_full:.3f} +/- {std_full:.3f} ms")

        isolated_results[vname] = {
            "forward_ms": mean_fwd, "forward_std": std_fwd,
            "backward_ms": mean_bwd, "backward_std": std_bwd,
            "full_ms": mean_full, "full_std": std_full,
        }

    # ── 6. End-to-end training iteration timing ──
    print(f"\n{'='*60}")
    print(f"Section 6: End-to-end Training Iteration ({N_MEASURE} measurements)")
    print(f"{'='*60}")

    def training_step(loss_fn):
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        pred = rendered[0].clamp(0, 1)
        loss = loss_fn(pred, gt0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    e2e_results = {}
    for vname, vfn in variants:
        print(f"\n  --- {vname} ---")
        # Warmup
        for _ in range(N_WARMUP):
            training_step(vfn)
        torch.cuda.synchronize()

        times = []
        for _ in range(N_MEASURE):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            training_step(vfn)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        arr = np.array(times)
        mean_e2e = float(arr.mean())
        std_e2e = float(arr.std())
        print(f"  End-to-end: {mean_e2e:.3f} +/- {std_e2e:.3f} ms")
        e2e_results[vname] = {"mean_ms": mean_e2e, "std_ms": std_e2e,
                              "min_ms": float(arr.min()), "max_ms": float(arr.max())}

    # ── 7. Memory measurement ──
    print(f"\n{'='*60}")
    print("Section 7: GPU Memory Allocation")
    print(f"{'='*60}")

    mem_results = {}
    for vname, vfn in variants:
        # Isolated fwd+bwd memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        pred = pred_base.clone().requires_grad_(True)
        loss = vfn(pred, gt0)
        loss.backward()
        torch.cuda.synchronize()
        peak_iso = torch.cuda.max_memory_allocated() / 1e6  # MB
        curr_iso = torch.cuda.memory_allocated() / 1e6

        # End-to-end memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        training_step(vfn)
        torch.cuda.synchronize()
        peak_e2e = torch.cuda.max_memory_allocated() / 1e6
        curr_e2e = torch.cuda.memory_allocated() / 1e6

        print(f"  {vname}: isolated peak={peak_iso:.1f}MB  e2e peak={peak_e2e:.1f}MB")
        mem_results[vname] = {"isolated_peak_MB": peak_iso, "isolated_current_MB": curr_iso,
                              "e2e_peak_MB": peak_e2e, "e2e_current_MB": curr_e2e}

    # ── 8. Profiler kernel count ──
    print(f"\n{'='*60}")
    print(f"Section 8: Profiler Kernel Count ({N_PROF} profiled iters)")
    print(f"{'='*60}")

    prof_results = {}
    trace_path_v2 = None
    for vname, vfn in variants:
        def iso_fn():
            pred = pred_base.clone().requires_grad_(True)
            loss = vfn(pred, gt0)
            loss.backward()
        kernels, n, tp = count_kernels(iso_fn)

        if vname == "V2":
            # Save V2 trace as the final trace
            final_trace = Path("results/phase-c42/c42_compile_trace.json")
            import shutil
            shutil.copy(str(tp), str(final_trace))
            trace_path_v2 = final_trace

        # Categorize
        cats = defaultdict(lambda: {"us": 0, "cnt": 0})
        for name, data in kernels.items():
            cat = categorize(name)
            cats[cat]["us"] += data["total_us"]
            cats[cat]["cnt"] += data["count"]

        total_us = sum(v["us"] for v in cats.values())
        total_cnt = sum(v["cnt"] for v in cats.values())

        print(f"\n  {vname}: {total_cnt/n:.1f} kernels/iter, {total_us/n/1000:.2f} ms/iter")
        for cat in sorted(cats.keys(), key=lambda c: cats[c]["us"], reverse=True):
            d = cats[cat]
            print(f"    {cat:<25s} {d['us']/n/1000:>8.3f} ms  {d['cnt']/n:>6.1f} calls")

        prof_results[vname] = {
            "total_kernels_per_iter": total_cnt / n,
            "total_ms_per_iter": total_us / n / 1000,
            "categories": {cat: {"ms": d["us"]/n/1000, "calls": d["cnt"]/n}
                          for cat, d in cats.items()},
        }

    # ── 9. Summary ──
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")

    v0_full = isolated_results["V0"]["full_ms"]
    v1_full = isolated_results["V1"]["full_ms"]
    v0_e2e = e2e_results["V0"]["mean_ms"]
    v1_e2e = e2e_results["V1"]["mean_ms"]

    print(f"\n  Isolated D-SSIM fwd+bwd:")
    print(f"    V0 (2D, no compile):        {v0_full:.3f} ms")
    print(f"    V1 (sep, no compile):       {v1_full:.3f} ms  ({v0_full/v1_full:.2f}x vs V0)")
    for vn in [k for k in isolated_results.keys() if k not in ("V0", "V1")]:
        v_full = isolated_results[vn]["full_ms"]
        print(f"    {vn} (sep + compile):        {v_full:.3f} ms  ({v0_full/v_full:.2f}x vs V0, {v1_full/v_full:.2f}x vs V1)")

    print(f"\n  End-to-end training iteration:")
    print(f"    V0: {v0_e2e:.3f} ms   V1: {v1_e2e:.3f} ms")
    e2e_speedup_v1 = (1 - v1_e2e / v0_e2e) * 100
    print(f"    V1 vs V0: {e2e_speedup_v1:+.1f}%")
    e2e_speedups = {}
    for vn in [k for k in e2e_results.keys() if k not in ("V0", "V1")]:
        v_e2e = e2e_results[vn]["mean_ms"]
        sp = (1 - v_e2e / v0_e2e) * 100
        e2e_speedups[vn] = sp
        print(f"    {vn} vs V0: {sp:+.1f}%  ({v_e2e:.3f} ms)")

    # Use best compiled variant for decision
    best_compiled = None
    best_e2e_speedup = -999
    for vn in e2e_speedups:
        if e2e_speedups[vn] > best_e2e_speedup:
            best_e2e_speedup = e2e_speedups[vn]
            best_compiled = vn

    if best_compiled:
        dssim_speedup_v2 = v0_full / isolated_results[best_compiled]["full_ms"]
    else:
        dssim_speedup_v2 = 0
        best_compiled = "N/A"

    print(f"\n  Best compiled variant: {best_compiled}")
    print(f"  D-SSIM speedup (V0 -> {best_compiled}): {dssim_speedup_v2:.2f}x")
    print(f"  E2E speedup (V0 -> {best_compiled}): {best_e2e_speedup:+.1f}%")

    print(f"\n  Correctness V0 vs {v2_name}: loss_diff={v0v2_loss_diff:.2e}  grad_diff={v0v2_grad_diff:.2e}")
    print(f"  Compilation time: {compile_time:.2f}s")

    print(f"\n  Memory (e2e peak):")
    for vn in isolated_results.keys():
        print(f"    {vn}: {mem_results[vn]['e2e_peak_MB']:.1f} MB")

    # Decision
    if dssim_speedup_v2 > 1.3 and best_e2e_speedup > 15:
        decision = "KEEP"
    elif dssim_speedup_v2 > 1.15 and best_e2e_speedup > 10:
        decision = "MAYBE"
    elif v0_full / v1_full > 1.3 and e2e_speedup_v1 > 15:
        decision = "MAYBE (V1 alone, no compile benefit)"
    else:
        decision = "DROP"
    print(f"\n  Decision: {decision}")

    # ── 10. Save JSON ──
    output = {
        "experiment": "C42 Phase 1-C: Separable Conv + torch.compile",
        "config": {"H": img_h, "W": img_w, "window": WINDOW, "sigma": SIGMA,
                   "n_warmup": N_WARMUP, "n_measure": N_MEASURE, "n_prof": N_PROF,
                   "lambda_dssim": LAMBDA_DSSIM,
                   "triton_available": False,
                   "v2_backend": "aot_eager" if V2_AVAILABLE else None,
                   "v2b_backend": "cudagraphs" if V2B_AVAILABLE else None},
        "correctness": {
            "V0_loss": correctness["V0"]["loss"],
            "V1_loss": correctness["V1"]["loss"],
            "V0_vs_v2_loss_diff": v0v2_loss_diff,
            "V0_vs_v2_grad_diff": v0v2_grad_diff,
            "gate_loss": 1e-5, "gate_grad": 1e-4,
            "passed": v0v2_loss_diff < 1e-5 and v0v2_grad_diff < 1e-4,
            "all_losses": {k: v["loss"] for k, v in correctness.items()},
        },
        "compilation_time_s": compile_time,
        "isolated_timing": isolated_results,
        "end_to_end_timing": e2e_results,
        "memory": mem_results,
        "profiler": prof_results,
        "speedup": {
            "dssim_v1_vs_v0": v0_full / v1_full,
            "dssim_best_compiled_vs_v0": dssim_speedup_v2,
            "e2e_v1_vs_v0_pct": e2e_speedup_v1,
            "e2e_best_compiled_vs_v0_pct": best_e2e_speedup,
            "best_compiled_variant": best_compiled,
        },
        "decision": decision,
        "trace_path": str(trace_path_v2) if trace_path_v2 else None,
    }

    save_path = Path("results/phase-c42/c42_compile_data.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")
    if trace_path_v2:
        print(f"  Trace saved to {trace_path_v2}")


if __name__ == "__main__":
    main()
