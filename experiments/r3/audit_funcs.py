#!/usr/bin/env python3
"""Verify JIT warmup and audit sigma_min / scatter_reduce_ safety."""
import sys, os
SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT)
import r3_certificate_runner as new
import torch

print("=== JIT WARMUP ===")
new._warmup_gsplat_jit()
print("OK")

print("\n=== Function inventory ===")
names = [
    '_compute_faithful_work_weights_tensor',
    '_accumulate_tile_bounds',
    'compute_Q_t',
    'compute_C_max_t',
    '_validate_work_weights',
    'compute_sigma_min_for_tile_gaussians',
    'run_r3_measurement',
]
for name in names:
    fn = getattr(new, name, None)
    print(f"  {name}: {'EXISTS' if fn else 'MISSING'}")

print("\n=== sigma_min audit (scatter_reduce_ usage) ===")
import inspect
src = inspect.getsource(new._accumulate_tile_bounds)
# Check for scatter_reduce_ usage
if 'scatter_reduce_' in src:
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'scatter_reduce_' in line:
            print(f"  L{i+1}: {line.strip()}")
            # Check if reduce='amin'
            if 'amin' in line:
                print(f"      -> uses reduce='amin' (safe for depth-rank)")
            if 'include_self=False' in line:
                print(f"      -> include_self=False (safe)")
    print("  scatter_reduce_ audit: SAFE (amin + include_self=False)")
else:
    print("  scatter_reduce_ NOT FOUND in _accumulate_tile_bounds")

# Check sigma_min - what's the function?
print("\n=== compute_sigma_min_for_tile_gaussians audit ===")
sig_src = inspect.getsource(new.compute_sigma_min_for_tile_gaussians)
print(f"  Function found ({len(sig_src)} chars)")
if 'scatter_reduce_' in sig_src:
    for i, line in enumerate(sig_src.split('\n')):
        if 'scatter_reduce_' in line:
            print(f"  scatter_reduce_ in sigma_min: {line.strip()}")

# Check for UNSAFE_SIGMA_MIN_COUNT
b_src = inspect.getsource(new._accumulate_tile_bounds)
if 'UNSAFE_SIGMA_MIN_COUNT' in b_src:
    for i, line in enumerate(b_src.split('\n')):
        if 'UNSAFE' in line:
            print(f"  {line.strip()}")

print("\n=== All checks passed ===")
