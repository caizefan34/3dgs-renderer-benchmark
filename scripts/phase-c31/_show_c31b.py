#!/usr/bin/env python3
"""Print camera isolation results."""
from __future__ import annotations
import json
from pathlib import Path

d = json.load(open(Path("results/phase-c31/c31_b_camera_isolate.json")))

print("=== C31-B Camera Isolation Results ===")
print()
for k in sorted(d["blocks"].keys()):
    blk = d["blocks"][k]
    p = blk["profiler"]
    print(f"  {k}:")
    print(f"    wall={blk['wall_ms']['mean']:.1f}+-{blk['wall_ms']['std']:.2f}ms")
    print(f"    fwd={blk['fwd_ms']['mean']:.2f}  bwd={blk['bwd_ms']['mean']:.2f}  opt={blk['opt_ms']['mean']:.2f}")
    print(f"    load={blk['cam_load_ms']['mean']:.3f}ms  peak_mem={blk['gpu_mem_peak_mb']:.0f}MB")
    print(f"    kernels/s={p['cuda_kernel_launches_per_step']}  gpu_busy={p['gpu_busy_ms_per_step']:.2f}ms")
    print(f"    gpu_gap={p['gpu_gap_ms_per_step_est']:.2f}ms")
    print()

print("Analysis:")
print(json.dumps(d["analysis"], indent=2))
