#!/usr/bin/env python3
"""Analyze Chrome trace from C32-A. Produces detailed breakdown of events."""
import json, sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase-c31/c32_a_trace2.json"

with open(path) as f:
    data = json.load(f)
events = data.get("traceEvents", data if isinstance(data, list) else [])
print(f"Total events: {len(events)}")

# Category distribution
cats = Counter(ev.get("cat", "?") for ev in events if isinstance(ev, dict))
print("\n=== Category distribution ===")
for cat, cnt in cats.most_common(30):
    print(f"  {cat}: {cnt}")

# CPU ops detail
cpu_ops = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "cpu_op"]
ops_by_name = Counter(ev["name"] for ev in cpu_ops if "name" in ev)
print(f"\n=== CPU ops ({len(cpu_ops)} total) ===")
for name, cnt in ops_by_name.most_common(40):
    print(f"  {name}: {cnt}")

# cuda_runtime detail
crt = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "cuda_runtime"]
crt_by_name = Counter(ev["name"] for ev in crt if "name" in ev)
print(f"\n=== CUDA runtime ({len(crt)} total) ===")
for name, cnt in crt_by_name.most_common(40):
    print(f"  {name}: {cnt}")

# GPU kernels
kernels = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "kernel"]
kn_by_name = Counter(ev["name"] for ev in kernels if "name" in ev)
print(f"\n=== GPU kernels ({len(kernels)} total) ===")
for name, cnt in kn_by_name.most_common(30):
    print(f"  {name[:120]}: {cnt}")

# Total GPU kernel time
gpu_total_us = sum(ev.get("dur", 0) for ev in kernels)
print(f"\nTotal GPU kernel time: {gpu_total_us/1000:.1f}ms over {len(kernels)} kernels")

# CPU ops total time
cpu_ops_total_us = sum(ev.get("dur", 0) for ev in cpu_ops)
print(f"Total CPU ops time: {cpu_ops_total_us/1000:.1f}ms over {len(cpu_ops)} ops")

# cuda_runtime total time
crt_total_us = sum(ev.get("dur", 0) for ev in crt)
print(f"Total CUDA runtime time: {crt_total_us/1000:.1f}ms over {len(crt)} calls")

# python_function total time
pyfn = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "python_function"]
pyfn_total_us = sum(ev.get("dur", 0) for ev in pyfn)
print(f"Total python_function time: {pyfn_total_us/1000:.1f}ms over {len(pyfn)} calls")

# gpu_memcpy and gpu_memset
memcpy = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "gpu_memcpy"]
memset = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "gpu_memset"]
memcpy_total_us = sum(ev.get("dur", 0) for ev in memcpy)
memset_total_us = sum(ev.get("dur", 0) for ev in memset)
print(f"\nGPU memcpy: {len(memcpy)} events, {memcpy_total_us/1000:.2f}ms total")
print(f"GPU memset: {len(memset)} events, {memset_total_us/1000:.2f}ms total")

# ac2g events (autograd graph)
ac2g = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "ac2g"]
if ac2g:
    ac2g_total_us = sum(ev.get("dur", 0) for ev in ac2g)
    print(f"\nac2g (autograd): {len(ac2g)} events, {ac2g_total_us/1000:.1f}ms total")

# fwdbwd events
fwdbwd = [ev for ev in events if isinstance(ev, dict) and ev.get("cat") == "fwdbwd"]
if fwdbwd:
    fwdbwd_total_us = sum(ev.get("dur", 0) for ev in fwdbwd)
    print(f"\nfwdbwd: {len(fwdbwd)} events, {fwdbwd_total_us/1000:.1f}ms total")
    for ev in fwdbwd[:10]:
        print(f"  {ev.get('name', '?')} dur={ev.get('dur', 0)}us")

# Look at specific kernel launch calls in cuda_runtime
launch_kernels = [ev for ev in crt if "launch" in ev.get("name", "").lower()]
print(f"\ncudaLaunch* calls: {len(launch_kernels)}")
launch_total_us = sum(ev.get("dur", 0) for ev in launch_kernels)
print(f"cudaLaunch* total time: {launch_total_us/1000:.3f}ms")

# Sync events
sync_events = [ev for ev in crt if "sync" in ev.get("name", "").lower()]
print(f"\ncuda*Synchronize calls: {len(sync_events)}")
sync_total_us = sum(ev.get("dur", 0) for ev in sync_events)
print(f"Sync total time: {sync_total_us/1000:.3f}ms")

# Memory events in cuda_runtime
malloc_events = [ev for ev in crt if any(x in ev.get("name","") for x in ["Malloc", "Free", "HostAlloc", "HostFree"])]
print(f"\nCUDA alloc/free calls: {len(malloc_events)}")
malloc_total_us = sum(ev.get("dur", 0) for ev in malloc_events)
print(f"Alloc/free total time: {malloc_total_us/1000:.3f}ms")
