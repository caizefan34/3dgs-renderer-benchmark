#!/usr/bin/env python3
"""Count exact elementwise operations in D-SSIM by tracing autograd graph."""
import sys, json
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from loss import d_ssim_loss, combined_loss

# Create small test tensors
H, W = 108, 192  # 10x downscale for quick test
pred = torch.randn(1, H, W, 3, device="cuda", requires_grad=True)
target = torch.randn(1, H, W, 3, device="cuda")

# Trace the forward pass with the profiler to count operations
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
) as prof:
    loss = combined_loss(pred, target, lambda_dssim=0.2)
    loss["loss"].backward()

# Export trace
prof.export_chrome_trace("results/phase-c31/c42_dssim_ops_trace.json")

# Count operations from key_averages
print("=== D-SSIM Autograd Graph Operations ===\n")

# Get key averages sorted by CUDA time
ka_list = prof.key_averages()
ka_sorted = sorted(ka_list, key=lambda x: getattr(x, 'self_cuda_time_total', 0) or getattr(x, 'cuda_time_total', 0), reverse=True)

print(f"{'Op Name':<60s} {'CPU us':>10s} {'CUDA us':>10s} {'Calls':>6s}")
print("-" * 90)
total_cuda = 0
total_calls = 0
for ka in ka_sorted:
    name = ka.key[:57] + "..." if len(ka.key) > 60 else ka.key
    cpu_us = ka.self_cpu_time_total
    # Try different attribute names for CUDA time
    cuda_us = getattr(ka, 'self_device_time_total', 0) or getattr(ka, 'cuda_time_total', 0) or 0
    count = ka.count
    total_cuda += cuda_us
    total_calls += count
    if cuda_us > 0 or cpu_us > 0:
        print(f"{name:<60s} {cpu_us:>10.1f} {cuda_us:>10.1f} {count:>6d}")

print(f"\nTotal CUDA time: {total_cuda/1000:.3f}ms")
print(f"Total op calls: {total_calls}")

# Count conv2d operations specifically
conv_ops = [ka for ka in ka_list if "conv" in ka.key.lower() or "cudnn" in ka.key.lower()]
print(f"\nConvolution operations: {len(conv_ops)}")
for ka in conv_ops:
    print(f"  {ka.key[:80]}: count={ka.count}")

# Count elementwise operations
elem_ops = [ka for ka in ka_list if "elementwise" in ka.key.lower() or "pointwise" in ka.key.lower()]
print(f"\nElementwise operations: {len(elem_ops)}")
total_elem_calls = sum(ka.count for ka in elem_ops)
print(f"  Total elementwise calls: {total_elem_calls}")

# Now do the full-size test to get exact kernel names
print("\n\n=== Full-size D-SSIM kernel attribution ===\n")
H2, W2 = 1080, 1920
pred_full = torch.randn(1, H2, W2, 3, device="cuda", requires_grad=True)
target_full = torch.randn(1, H2, W2, 3, device="cuda")

# Warmup
for _ in range(3):
    p = torch.randn(1, H2, W2, 3, device="cuda", requires_grad=True)
    l = combined_loss(p, target_full, lambda_dssim=0.2)
    l["loss"].backward()

# Profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
) as prof2:
    for i in range(8):
        p = torch.randn(1, H2, W2, 3, device="cuda", requires_grad=True)
        l = combined_loss(p, target_full, lambda_dssim=0.2)
        l["loss"].backward()
        prof2.step()

prof2.export_chrome_trace("results/phase-c31/c42_dssim_full_trace.json")

# Parse trace
with open("results/phase-c31/c42_dssim_full_trace.json") as f:
    trace = json.load(f)

from collections import defaultdict

kernels = defaultdict(lambda: {"total_us": 0, "count": 0})
for evt in trace.get("traceEvents", []):
    if evt.get("cat") == "kernel" and evt.get("dur", 0) > 0:
        name = evt["name"]
        kernels[name]["total_us"] += evt["dur"]
        kernels[name]["count"] += 1

N_PROF = 5
print(f"Full-size D-SSIM-only kernels (5 profiled iters, {H2}x{W2}x3):")
print(f"{'Kernel':<90s} {'ms/iter':>10s} {'Calls/iter':>10s}")
print("-" * 115)
total = 0
for name, data in sorted(kernels.items(), key=lambda x: x[1]["total_us"], reverse=True)[:20]:
    short = name[:87] + "..." if len(name) > 90 else name
    ms_iter = data["total_us"] / N_PROF / 1000
    calls = data["count"] / N_PROF
    total += ms_iter
    print(f"{short:<90s} {ms_iter:>10.3f} {calls:>10.1f}")
print(f"\nTotal: {total:.3f} ms/iter")
