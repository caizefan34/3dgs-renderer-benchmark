#!/usr/bin/env python3
"""Extract exact D-SSIM kernel names and durations from C41 trace."""
import json
from collections import defaultdict

with open("results/phase-c31/c41_trace.json") as f:
    trace = json.load(f)

# Find all kernel events
kernels = []
for evt in trace.get("traceEvents", []):
    if evt.get("cat") == "kernel" and evt.get("dur", 0) > 0:
        kernels.append({
            "name": evt["name"],
            "dur_us": evt["dur"],
            "ts": evt.get("ts", 0),
        })

# Categorize D-SSIM kernels (cuDNN convolution + tensorTransform)
dssim_keywords = ["cudnn", "cutlass", "tensorTransform", "dgrad", "wgrad", "conv2d", "fgrad"]
dssim_kernels = [k for k in kernels if any(kw in k["name"].lower() for kw in dssim_keywords)]

print(f"Total kernel events: {len(kernels)}")
print(f"D-SSIM kernel events: {len(dssim_kernels)}")
print()

# Aggregate by unique kernel name
name_totals = defaultdict(lambda: {"total_us": 0, "count": 0, "durations": []})
for k in dssim_kernels:
    name_totals[k["name"]]["total_us"] += k["dur_us"]
    name_totals[k["name"]]["count"] += 1
    name_totals[k["name"]]["durations"].append(k["dur_us"])

# Sort by total time
N_PROF = 10  # profiled iterations
print("D-SSIM kernels (sorted by total time, 10 profiled iters):")
print(f"{'Kernel':<100s} {'Total ms':>10s} {'ms/iter':>10s} {'Count':>6s} {'Calls/iter':>10s} {'Mean us':>10s}")
print("-" * 150)

for name, data in sorted(name_totals.items(), key=lambda x: x[1]["total_us"], reverse=True):
    short = name[:97] + "..." if len(name) > 100 else name
    total_ms = data["total_us"] / 1000
    per_iter = total_ms / N_PROF
    calls_per_iter = data["count"] / N_PROF
    mean_us = data["total_us"] / data["count"]
    print(f"{short:<100s} {total_ms:>10.3f} {per_iter:>10.3f} {data['count']:>6d} {calls_per_iter:>10.1f} {mean_us:>10.1f}")

total_dssim_ms = sum(v["total_us"] for v in name_totals.values()) / 1000
print(f"\nTotal D-SSIM kernel time: {total_dssim_ms:.1f}ms ({total_dssim_ms/N_PROF:.1f}ms/iter)")

# Also show all kernel name categories
print(f"\n\nAll kernel categories (first 20 chars of name):")
cat_totals = defaultdict(lambda: {"total_us": 0, "count": 0})
for k in kernels:
    # Extract a short category from the name
    name = k["name"]
    if "cudnn" in name.lower() or "cutlass" in name.lower() or "tensorTransform" in name:
        cat = "cudnn/cutlass (D-SSIM)"
    elif "gsplat" in name.lower():
        if "bwd" in name.lower():
            cat = "gsplat_bwd"
        else:
            cat = "gsplat_fwd"
    elif "multi_tensor" in name.lower():
        cat = "adam (multi_tensor)"
    elif "elementwise" in name.lower() or "pointwise" in name.lower():
        cat = "elementwise"
    elif "vectorized" in name.lower():
        cat = "vectorized_elementwise"
    elif "radixsort" in name.lower() or "cub" in name.lower():
        cat = "sorting (cub)"
    elif "fill" in name.lower():
        cat = "fill"
    elif "reduce" in name.lower():
        cat = "reduction"
    elif "memcpy" in name.lower() or "memset" in name.lower():
        cat = "memcpy/memset"
    else:
        cat = f"other ({name[:30]})"
    cat_totals[cat]["total_us"] += k["dur_us"]
    cat_totals[cat]["count"] += 1

print(f"{'Category':<40s} {'ms/iter':>10s} {'%':>7s} {'Kernels':>8s} {'K/iter':>8s}")
print("-" * 80)
total_all = sum(v["total_us"] for v in cat_totals.values())
for cat, data in sorted(cat_totals.items(), key=lambda x: x[1]["total_us"], reverse=True):
    ms_iter = data["total_us"] / N_PROF / 1000
    pct = data["total_us"] / total_all * 100
    k_per_iter = data["count"] / N_PROF
    print(f"{cat:<40s} {ms_iter:>10.3f} {pct:>6.1f}% {data['count']:>8d} {k_per_iter:>8.1f}")
