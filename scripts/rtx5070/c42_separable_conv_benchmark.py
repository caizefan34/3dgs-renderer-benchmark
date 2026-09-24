#!/usr/bin/env python3
"""
C42 Phase 1-B: Separable Gaussian Convolution Validation.

Isolated benchmark — does NOT modify training pipeline.

Experiments:
  A. Numerical correctness: 2D vs separable blur
  B. Forward benchmark: 2D conv vs 1D+1D conv
  C. Backward benchmark: dgrad for both
"""
import json, math, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# ── Setup ──
DEVICE = "cuda"
DTYPE = torch.float32
H, W = 1080, 1920
WINDOW = 11
SIGMA = 1.5
N_WARMUP = 20
N_MEASURE = 50
N_PROF = 10


def build_kernel_2d():
    """Build 2D Gaussian kernel [3,1,11,11] — same as loss.py lines 34-38."""
    coords = torch.arange(WINDOW, device=DEVICE, dtype=DTYPE) - WINDOW // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * SIGMA ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]          # [11,11]
    return kernel_2d.expand(3, 1, WINDOW, WINDOW).contiguous()   # [3,1,11,11]


def build_kernel_separable():
    """Build separable 1D kernels [3,1,1,11] and [3,1,11,1]."""
    coords = torch.arange(WINDOW, device=DEVICE, dtype=DTYPE) - WINDOW // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * SIGMA ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    # Horizontal: [3,1,1,11] — convolve along width
    kh = kernel_1d.view(1, 1, 1, WINDOW).expand(3, 1, 1, WINDOW).contiguous()
    # Vertical: [3,1,11,1] — convolve along height
    kv = kernel_1d.view(1, 1, WINDOW, 1).expand(3, 1, WINDOW, 1).contiguous()
    return kh, kv


def blur_2d(x, kernel_2d):
    """Original 2D depthwise convolution — matches loss.py line 44."""
    return F.conv2d(x, kernel_2d, padding=WINDOW // 2, groups=3)


def blur_separable(x, kh, kv):
    """Separable: horizontal then vertical 1D convolutions."""
    x_h = F.conv2d(x, kh, padding=(0, WINDOW // 2), groups=3)
    return F.conv2d(x_h, kv, padding=(WINDOW // 2, 0), groups=3)


def cuda_event_time(fn, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    """Measure CUDA event latency of a function. Returns (mean_ms, std_ms, min_ms, max_ms)."""
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


def count_kernels(fn, n_iters=N_PROF):
    """Run profiler, return per-kernel-name aggregated durations (us) and total count."""
    # Warmup
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

    trace_path = Path("results/phase-c42/c42_b_trace_tmp.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    with open(trace_path) as f:
        trace = json.load(f)

    kernels = defaultdict(lambda: {"total_us": 0, "count": 0})
    for evt in trace.get("traceEvents", []):
        if evt.get("cat") == "kernel" and evt.get("dur", 0) > 0:
            kernels[evt["name"]]["total_us"] += evt["dur"]
            kernels[evt["name"]]["count"] += 1

    return kernels, n_iters


def fmt_short(name, maxlen=75):
    if len(name) > maxlen:
        return name[:maxlen-3] + "..."
    return name


# ═══════════════════════════════════════════════════════════════
# Experiment A: Numerical Correctness
# ═══════════════════════════════════════════════════════════════
print("=" * 72)
print("Experiment A: Numerical Correctness (2D vs Separable)")
print("=" * 72)

torch.manual_seed(42)
x_test = torch.randn(1, 3, H, W, device=DEVICE, dtype=DTYPE)

k2d = build_kernel_2d()
kh, kv = build_kernel_separable()

result_2d = blur_2d(x_test, k2d)
result_sep = blur_separable(x_test, kh, kv)

diff = (result_2d - result_sep).abs()
max_abs_err = float(diff.max().item())
mean_abs_err = float(diff.mean().item())
rel_err = float((diff / (result_2d.abs() + 1e-8)).mean().item())

# Also check with realistic image values [0,1]
x_img = torch.rand(1, 3, H, W, device=DEVICE, dtype=DTYPE)
r2d_img = blur_2d(x_img, k2d)
rsep_img = blur_separable(x_img, kh, kv)
diff_img = (r2d_img - rsep_img).abs()
max_abs_err_img = float(diff_img.max().item())
mean_abs_err_img = float(diff_img.mean().item())

print(f"  Input: [1,3,{H},{W}] float32")
print(f"  Random data:  max_abs_error={max_abs_err:.2e}  mean_abs_error={mean_abs_err:.2e}  rel_error={rel_err:.2e}")
print(f"  Image [0,1]:  max_abs_error={max_abs_err_img:.2e}  mean_abs_error={mean_abs_err_img:.2e}")
print(f"  Gate: max_abs_error < 1e-5")
print(f"  Result: {'PASS' if max(max_abs_err, max_abs_err_img) < 1e-5 else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════
# Experiment B: Forward Benchmark
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print(f"Experiment B: Forward Benchmark ({N_MEASURE} measurements)")
print(f"{'=' * 72}")

x_fwd = torch.randn(1, 3, H, W, device=DEVICE, dtype=DTYPE)

# B1: Single blur call — 2D
mean_2d, std_2d, min_2d, max_2d = cuda_event_time(lambda: blur_2d(x_fwd, k2d))
print(f"\n  B1: Single blur — 2D conv [3,1,11,11]")
print(f"      mean={mean_2d:.3f}ms  std={std_2d:.3f}ms  min={min_2d:.3f}ms  max={max_2d:.3f}ms")

# B2: Single blur call — separable
mean_sep, std_sep, min_sep, max_sep = cuda_event_time(lambda: blur_separable(x_fwd, kh, kv))
print(f"\n  B2: Single blur — separable [3,1,1,11]+[3,1,11,1]")
print(f"      mean={mean_sep:.3f}ms  std={std_sep:.3f}ms  min={min_sep:.3f}ms  max={max_sep:.3f}ms")

fwd_speedup = mean_2d / mean_sep
print(f"\n  Forward speedup: {fwd_speedup:.2f}×  ({(1-1/fwd_speedup)*100:.1f}% reduction)")

# B3: Full D-SSIM forward (5 blur calls) — 2D
def dssim_forward_2d(pred, tgt, kernel):
    mu_p = blur_2d(pred, kernel)
    mu_t = blur_2d(tgt, kernel)
    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t
    sig_p = blur_2d(pred ** 2, kernel) - mu_p_sq
    sig_t = blur_2d(tgt ** 2, kernel) - mu_t_sq
    sig_pt = blur_2d(pred * tgt, kernel) - mu_pt
    C1, C2 = 0.0001, 0.0009
    ssim = ((2*mu_pt + C1) * (2*sig_pt + C2)) / ((mu_p_sq + mu_t_sq + C1) * (sig_p + sig_t + C2))
    return 1.0 - ssim.mean()

def dssim_forward_sep(pred, tgt, kh, kv):
    mu_p = blur_separable(pred, kh, kv)
    mu_t = blur_separable(tgt, kh, kv)
    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t
    sig_p = blur_separable(pred ** 2, kh, kv) - mu_p_sq
    sig_t = blur_separable(tgt ** 2, kh, kv) - mu_t_sq
    sig_pt = blur_separable(pred * tgt, kh, kv) - mu_pt
    C1, C2 = 0.0001, 0.0009
    ssim = ((2*mu_pt + C1) * (2*sig_pt + C2)) / ((mu_p_sq + mu_t_sq + C1) * (sig_p + sig_t + C2))
    return 1.0 - ssim.mean()

pred_fwd = torch.rand(1, 3, H, W, device=DEVICE, dtype=DTYPE)
tgt_fwd = torch.rand(1, 3, H, W, device=DEVICE, dtype=DTYPE)

mean_dssim_2d, std_dssim_2d, _, _ = cuda_event_time(lambda: dssim_forward_2d(pred_fwd, tgt_fwd, k2d))
mean_dssim_sep, std_dssim_sep, _, _ = cuda_event_time(lambda: dssim_forward_sep(pred_fwd, tgt_fwd, kh, kv))

print(f"\n  B3: Full D-SSIM forward (5 blur calls + elementwise)")
print(f"      2D:         mean={mean_dssim_2d:.3f}ms  std={std_dssim_2d:.3f}ms")
print(f"      Separable:  mean={mean_dssim_sep:.3f}ms  std={std_dssim_sep:.3f}ms")
dssim_fwd_speedup = mean_dssim_2d / mean_dssim_sep
print(f"      D-SSIM forward speedup: {dssim_fwd_speedup:.2f}×  ({(1-1/dssim_fwd_speedup)*100:.1f}% reduction)")

# B4: Profiler kernel count — single blur
print(f"\n  B4: Profiler kernel count — single blur ({N_PROF} profiled iters)")
kernels_2d_blur, n1 = count_kernels(lambda: blur_2d(x_fwd, k2d))
kernels_sep_blur, n2 = count_kernels(lambda: blur_separable(x_fwd, kh, kv))

print(f"      {'Kernel':<75s} {'ms/iter':>8s} {'calls/iter':>10s}")
print(f"      {'-'*75} {'-'*8} {'-'*10}")
print(f"      2D conv:")
total_2d = 0
for name, data in sorted(kernels_2d_blur.items(), key=lambda x: x[1]["total_us"], reverse=True):
    ms = data["total_us"] / n1 / 1000
    total_2d += ms
    print(f"        {fmt_short(name):<75s} {ms:>8.3f} {data['count']/n1:>10.1f}")
print(f"        {'TOTAL':<75s} {total_2d:>8.3f} {sum(v['count'] for v in kernels_2d_blur.values())/n1:>10.1f}")

print(f"      Separable:")
total_sep = 0
for name, data in sorted(kernels_sep_blur.items(), key=lambda x: x[1]["total_us"], reverse=True):
    ms = data["total_us"] / n2 / 1000
    total_sep += ms
    print(f"        {fmt_short(name):<75s} {ms:>8.3f} {data['count']/n2:>10.1f}")
print(f"        {'TOTAL':<75s} {total_sep:>8.3f} {sum(v['count'] for v in kernels_sep_blur.values())/n2:>10.1f}")


# ═══════════════════════════════════════════════════════════════
# Experiment C: Backward Benchmark
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print(f"Experiment C: Backward Benchmark ({N_MEASURE} measurements)")
print(f"{'=' * 72}")

# C1: Single blur backward — 2D
def blur_2d_bwd():
    x = torch.randn(1, 3, H, W, device=DEVICE, dtype=DTYPE, requires_grad=True)
    out = blur_2d(x, k2d)
    out.mean().backward()

def blur_sep_bwd():
    x = torch.randn(1, 3, H, W, device=DEVICE, dtype=DTYPE, requires_grad=True)
    out = blur_separable(x, kh, kv)
    out.mean().backward()

mean_2d_bwd, std_2d_bwd, _, _ = cuda_event_time(blur_2d_bwd)
mean_sep_bwd, std_sep_bwd, _, _ = cuda_event_time(blur_sep_bwd)

print(f"\n  C1: Single blur backward (fwd+bwd)")
print(f"      2D:         mean={mean_2d_bwd:.3f}ms  std={std_2d_bwd:.3f}ms")
print(f"      Separable:  mean={mean_sep_bwd:.3f}ms  std={std_sep_bwd:.3f}ms")
bwd_speedup_single = mean_2d_bwd / mean_sep_bwd
print(f"      Speedup: {bwd_speedup_single:.2f}×")

# C2: Full D-SSIM backward — 2D (3 pred-dependent blur calls have grad)
def dssim_bwd_2d():
    pred = torch.rand(1, 3, H, W, device=DEVICE, dtype=DTYPE, requires_grad=True)
    tgt = torch.rand(1, 3, H, W, device=DEVICE, dtype=DTYPE)
    loss = dssim_forward_2d(pred, tgt, k2d)
    loss.backward()

def dssim_bwd_sep():
    pred = torch.rand(1, 3, H, W, device=DEVICE, dtype=DTYPE, requires_grad=True)
    tgt = torch.rand(1, 3, H, W, device=DEVICE, dtype=DTYPE)
    loss = dssim_forward_sep(pred, tgt, kh, kv)
    loss.backward()

mean_dssim_2d_bwd, std_dssim_2d_bwd, _, _ = cuda_event_time(dssim_bwd_2d)
mean_dssim_sep_bwd, std_dssim_sep_bwd, _, _ = cuda_event_time(dssim_bwd_sep)

print(f"\n  C2: Full D-SSIM fwd+bwd (3 grad blur + 2 no-grad blur + elementwise)")
print(f"      2D:         mean={mean_dssim_2d_bwd:.3f}ms  std={std_dssim_2d_bwd:.3f}ms")
print(f"      Separable:  mean={mean_dssim_sep_bwd:.3f}ms  std={std_dssim_sep_bwd:.3f}ms")
dssim_bwd_speedup = mean_dssim_2d_bwd / mean_dssim_sep_bwd
print(f"      D-SSIM fwd+bwd speedup: {dssim_bwd_speedup:.2f}×  ({(1-1/dssim_bwd_speedup)*100:.1f}% reduction)")

# C3: Profiler kernel count — full D-SSIM fwd+bwd
print(f"\n  C3: Profiler kernel count — full D-SSIM fwd+bwd ({N_PROF} profiled iters)")
kernels_2d_full, n3 = count_kernels(dssim_bwd_2d)
kernels_sep_full, n4 = count_kernels(dssim_bwd_sep)

# Categorize
def categorize(name):
    nl = name.lower()
    if "dgrad" in nl:
        return "dgrad (backward conv)"
    elif "conv2d" in nl:
        return "conv2d (forward conv)"
    elif "elementwise" in nl or "vectorized" in nl or "pow" in nl or "neg" in nl:
        return "elementwise"
    elif "reduce" in nl:
        return "reduction"
    else:
        return "other"

print(f"      {'Category':<30s} {'2D ms/iter':>12s} {'2D calls':>10s} {'Sep ms/iter':>12s} {'Sep calls':>10s}")
print(f"      {'-'*30} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

cats_2d = defaultdict(lambda: {"us": 0, "cnt": 0})
cats_sep = defaultdict(lambda: {"us": 0, "cnt": 0})
for name, data in kernels_2d_full.items():
    cat = categorize(name)
    cats_2d[cat]["us"] += data["total_us"]
    cats_2d[cat]["cnt"] += data["count"]
for name, data in kernels_sep_full.items():
    cat = categorize(name)
    cats_sep[cat]["us"] += data["total_us"]
    cats_sep[cat]["cnt"] += data["count"]

all_cats = sorted(set(list(cats_2d.keys()) + list(cats_sep.keys())),
                  key=lambda c: cats_2d.get(c, {"us": 0})["us"], reverse=True)
for cat in all_cats:
    d2 = cats_2d.get(cat, {"us": 0, "cnt": 0})
    ds = cats_sep.get(cat, {"us": 0, "cnt": 0})
    print(f"      {cat:<30s} {d2['us']/n3/1000:>12.3f} {d2['cnt']/n3:>10.1f} {ds['us']/n4/1000:>12.3f} {ds['cnt']/n4:>10.1f}")

total_2d_us = sum(v["us"] for v in cats_2d.values())
total_sep_us = sum(v["us"] for v in cats_sep.values())
total_2d_cnt = sum(v["cnt"] for v in cats_2d.values())
total_sep_cnt = sum(v["cnt"] for v in cats_sep.values())
print(f"      {'TOTAL':<30s} {total_2d_us/n3/1000:>12.3f} {total_2d_cnt/n3:>10.1f} {total_sep_us/n4/1000:>12.3f} {total_sep_cnt/n4:>10.1f}")

# Show individual kernel names for conv types
print(f"\n      Convolution kernels (2D):")
for name, data in sorted(kernels_2d_full.items(), key=lambda x: x[1]["total_us"], reverse=True):
    if "conv" in name.lower() or "dgrad" in name.lower():
        print(f"        {fmt_short(name, 70):<70s} {data['total_us']/n3/1000:>8.3f}ms  {data['count']/n3:.1f} calls")

print(f"\n      Convolution kernels (Separable):")
for name, data in sorted(kernels_sep_full.items(), key=lambda x: x[1]["total_us"], reverse=True):
    if "conv" in name.lower() or "dgrad" in name.lower():
        print(f"        {fmt_short(name, 70):<70s} {data['total_us']/n4/1000:>8.3f}ms  {data['count']/n4:.1f} calls")


# ═══════════════════════════════════════════════════════════════
# Summary & Save
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("SUMMARY")
print(f"{'=' * 72}")
print(f"  Correctness:     max_abs_error={max(max_abs_err, max_abs_err_img):.2e}  gate<1e-5  → {'PASS' if max(max_abs_err, max_abs_err_img) < 1e-5 else 'FAIL'}")
print(f"  Single blur fwd: {fwd_speedup:.2f}×  ({mean_2d:.2f}ms → {mean_sep:.2f}ms)")
print(f"  Single blur bwd: {bwd_speedup_single:.2f}×  ({mean_2d_bwd:.2f}ms → {mean_sep_bwd:.2f}ms)")
print(f"  D-SSIM fwd only: {dssim_fwd_speedup:.2f}×  ({mean_dssim_2d:.2f}ms → {mean_dssim_sep:.2f}ms)")
print(f"  D-SSIM fwd+bwd:  {dssim_bwd_speedup:.2f}×  ({mean_dssim_2d_bwd:.2f}ms → {mean_dssim_sep_bwd:.2f}ms)")

# Estimate impact on full training iteration
# C41 measured D-SSIM = 38.0ms/iter in pipeline (profiler), total = 89.5ms/iter
# Isolated measurement may differ; use ratio
c41_dssim_ms = 38.0
c41_total_ms = 89.5
est_new_dssim = c41_dssim_ms / dssim_bwd_speedup
est_total = c41_total_ms - c41_dssim_ms + est_new_dssim
print(f"\n  Estimated impact on training pipeline (from C41: D-SSIM={c41_dssim_ms}ms, total={c41_total_ms}ms):")
print(f"    New D-SSIM time:  {est_new_dssim:.1f}ms  (was {c41_dssim_ms}ms)")
print(f"    New total time:   {est_total:.1f}ms  (was {c41_total_ms}ms)")
print(f"    Total speedup:    {(1 - est_total/c41_total_ms)*100:.1f}%")

# Decision
if max(max_abs_err, max_abs_err_img) >= 1e-5:
    decision = "DROP — numerical correctness failed"
elif dssim_bwd_speedup >= 2.0:
    decision = "KEEP — significant speedup with identical math"
elif dssim_bwd_speedup >= 1.3:
    decision = "MAYBE — moderate speedup, evaluate in full pipeline"
else:
    decision = "DROP — insufficient speedup"

print(f"\n  Decision: {decision}")

# Save JSON
output = {
    "experiment": "C42 Phase 1-B: Separable Convolution Validation",
    "config": {"H": H, "W": W, "window": WINDOW, "sigma": SIGMA,
               "n_warmup": N_WARMUP, "n_measure": N_MEASURE, "n_prof": N_PROF},
    "A_correctness": {
        "max_abs_error_random": max_abs_err,
        "mean_abs_error_random": mean_abs_err,
        "max_abs_error_image": max_abs_err_img,
        "mean_abs_error_image": mean_abs_err_img,
        "gate": 1e-5,
        "passed": max(max_abs_err, max_abs_err_img) < 1e-5,
    },
    "B_forward": {
        "single_blur_2d_ms": mean_2d,
        "single_blur_sep_ms": mean_sep,
        "single_blur_speedup": fwd_speedup,
        "dssim_fwd_2d_ms": mean_dssim_2d,
        "dssim_fwd_sep_ms": mean_dssim_sep,
        "dssim_fwd_speedup": dssim_fwd_speedup,
        "kernel_count_2d": {name: {"ms": d["total_us"]/n1/1000, "calls": d["count"]/n1}
                           for name, d in kernels_2d_blur.items()},
        "kernel_count_sep": {name: {"ms": d["total_us"]/n2/1000, "calls": d["count"]/n2}
                           for name, d in kernels_sep_blur.items()},
    },
    "C_backward": {
        "single_blur_bwd_2d_ms": mean_2d_bwd,
        "single_blur_bwd_sep_ms": mean_sep_bwd,
        "single_blur_bwd_speedup": bwd_speedup_single,
        "dssim_bwd_2d_ms": mean_dssim_2d_bwd,
        "dssim_bwd_sep_ms": mean_dssim_sep_bwd,
        "dssim_bwd_speedup": dssim_bwd_speedup,
        "categories_2d": {cat: {"ms": d["us"]/n3/1000, "calls": d["cnt"]/n3}
                         for cat, d in cats_2d.items()},
        "categories_sep": {cat: {"ms": d["us"]/n4/1000, "calls": d["cnt"]/n4}
                          for cat, d in cats_sep.items()},
    },
    "estimated_pipeline_impact": {
        "c41_dssim_ms": c41_dssim_ms,
        "c41_total_ms": c41_total_ms,
        "est_new_dssim_ms": est_new_dssim,
        "est_new_total_ms": est_total,
        "total_speedup_pct": (1 - est_total/c41_total_ms)*100,
    },
    "decision": decision,
}

save_path = Path("results/phase-c42/c42_separable_conv_data.json")
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Data saved to {save_path}")
