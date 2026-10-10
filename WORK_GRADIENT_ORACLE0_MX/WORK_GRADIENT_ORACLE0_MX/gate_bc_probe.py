#!/usr/bin/env python3
"""Gate B/C: verify rasterize_gaussian_higs_dynamic exists, accepts tile_mask,
and the native isect_tiles CUDA op supports tile_mask sparse execution."""
import sys, inspect, traceback

print("=== GATE B: rasterize_gaussian_higs_dynamic ===")
from gsplat.experimental import render
from gsplat.experimental.render.functional import gaussian_inference as gi

fn = gi.rasterize_gaussian_higs_dynamic
print("function:", fn)
print("signature:", str(inspect.signature(fn)))
params = inspect.signature(fn).parameters
print("has tile_mask param:", "tile_mask" in params)
print("has tile_sampling_ratio param:", "tile_sampling_ratio" in params)

# Check the native isect_tiles supports tile_mask
print("\n=== isect_tiles native op signature ===")
try:
    from gsplat.cuda._wrapper import isect_tiles
    sig = inspect.signature(isect_tiles)
    print("isect_tiles signature:", sig)
    print("has tile_mask:", "tile_mask" in sig.parameters)
except Exception as e:
    print("isect_tiles import/inspect failed:", e)
    traceback.print_exc()

# Check torch ops for higs scene cuda
print("\n=== torch.ops namespace ===")
try:
    import torch
    print("has torch.ops.experimental:", hasattr(torch.ops, "experimental"))
except Exception as e:
    print("torch.ops check:", e)

# Try to access the scene_cuda extension directly by adding build dir to path
print("\n=== try load gsplat_scene_cuda from build dir ===")
import os, glob
build_so = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/gsplat_scene_cuda.so"
print("build .so exists:", os.path.exists(build_so))
if os.path.exists(build_so):
    import importlib.util
    spec = importlib.util.spec_from_file_location("gsplat_scene_cuda_test", build_so)
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("LOADED scene_cuda .so OK")
        print("attrs:", [a for a in dir(mod) if not a.startswith('__')][:40])
    except Exception as e:
        print("scene_cuda .so load FAILED:", e)
        traceback.print_exc()
