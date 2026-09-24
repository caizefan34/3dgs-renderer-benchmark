#!/usr/bin/env python3
"""Precise decomposition of CUDA runtime events by type."""
import json, sys
import numpy as np
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase-c31/c32_a_trace2.json"

with open(path) as f:
    data = json.load(f)
events = data.get("traceEvents", data if isinstance(data, list) else [])

# Filter cuda_runtime events
crt = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "cuda_runtime"]
print(f"CUDA runtime events total: {len(crt)} over 3 profiled iterations")

# Group by name
by_name = defaultdict(list)
for ev in crt:
    by_name[ev.get("name", "?")].append(ev.get("dur", 0))

print(f"\n{'Call':<60} {'Count':>6} {'Total(ms)':>10} {'Mean(us)':>9} {'P50(us)':>8} {'P90(us)':>8} {'P99(us)':>8}")
print("-" * 110)
for name in sorted(by_name.keys(), key=lambda n: -sum(by_name[n])):
    durs = np.array(by_name[name])
    total_ms = np.sum(durs) / 1000
    mean_us = np.mean(durs)
    p50 = np.percentile(durs, 50)
    p90 = np.percentile(durs, 90)
    p99 = np.percentile(durs, 99)
    print(f"{name:<60} {len(durs):>6} {total_ms:>9.2f} {mean_us:>8.1f} {p50:>7.1f} {p90:>7.1f} {p99:>7.1f}")

# Per-iteration estimate (3 profiled iters)
print(f"\n--- Per-iteration (÷3) ---")
for name in sorted(by_name.keys(), key=lambda n: -sum(by_name[n])):
    durs = np.array(by_name[name])
    total_ms = np.sum(durs) / 1000
    per_iter_ms = total_ms / 3
    if per_iter_ms > 0.1:
        print(f"  {name}: {per_iter_ms:.3f}ms/iter ({len(durs)//3} calls/iter)")

# Key finding: cudaLaunchKernel
lk = by_name.get("cudaLaunchKernel", [0])
if lk:
    print(f"\ncudaLaunchKernel: {len(lk)} calls over 3 iters = {len(lk)//3} calls/iter")
    print(f"  Total CPU time: {np.sum(lk)/1000:.3f}ms over 3 iters")
    print(f"  Per iteration: {np.sum(lk)/1000/3:.3f}ms")

# Sum ALL non-launch CUDA runtime calls
non_launch_ms = sum(sum(v) for k, v in by_name.items() if k != "cudaLaunchKernel") / 1000
launch_ms = sum(by_name.get("cudaLaunchKernel", [0])) / 1000
print(f"\nAll CUDA runtime calls: {launch_ms + non_launch_ms:.2f}ms over 3 iters")
print(f"  cudaLaunchKernel:      {launch_ms:.2f}ms ({launch_ms/(launch_ms+non_launch_ms)*100:.1f}%)")
print(f"  Non-launch overhead:   {non_launch_ms:.2f}ms ({(non_launch_ms/(launch_ms+non_launch_ms))*100:.1f}%)")

# Sync calls detailed
sync_calls = {k: v for k, v in by_name.items() if "ync" in k or "Synchronize" in k}
if sync_calls:
    print("\nSynchronization calls:")
    for name, durs in sorted(sync_calls.items()):
        print(f"  {name}: {len(durs)} calls, {np.sum(durs)/1000:.3f}ms total")

# Memory calls
mem_calls = {k: v for k, v in by_name.items() if "Memcpy" in k or "Memset" in k or "Malloc" in k or "Free" in k}
if mem_calls:
    print("\nMemory calls:")
    for name, durs in sorted(mem_calls.items()):
        print(f"  {name}: {len(durs)} calls, {np.sum(durs)/1000:.3f}ms total")

# Also look at cpu_op events for aten ops timing
cpu_ops = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "cpu_op"]
aten_empty = [ev.get("dur",0) for ev in cpu_ops if ev.get("name","") == "aten::empty"]
aten_zero = [ev.get("dur",0) for ev in cpu_ops if ev.get("name","") == "aten::zero_"]
aten_fill = [ev.get("dur",0) for ev in cpu_ops if ev.get("name","") == "aten::fill_"]
aten_copy = [ev.get("dur",0) for ev in cpu_ops if ev.get("name","") == "aten::copy_"]

print(f"\nKey CPU ops:")
for label, lst in [("aten::empty", aten_empty), ("aten::zero_", aten_zero), 
                    ("aten::fill_", aten_fill), ("aten::copy_", aten_copy)]:
    if lst:
        print(f"  {label}: {len(lst)} calls, {np.sum(lst)/1000:.3f}ms total, mean={np.mean(lst):.1f}us")
