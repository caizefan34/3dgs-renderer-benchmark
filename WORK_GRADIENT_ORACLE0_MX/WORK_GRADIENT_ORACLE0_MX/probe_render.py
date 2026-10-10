#!/usr/bin/env python3
"""Deep probe: how does experimental.render load its CUDA extension?"""
import sys, os, json, hashlib, importlib

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

print("=== experimental.render import chain ===")
try:
    from gsplat.experimental import render
    print("render module:", render.__file__)
    print("render attrs:", [a for a in dir(render) if not a.startswith('__')][:30])
except Exception as e:
    import traceback; traceback.print_exc()
    print("RENDER_FAIL:", e)

# Try to import the functional module
print("\n=== functional.gaussian_inference ===")
try:
    from gsplat.experimental.render.functional import gaussian_inference as gi
    print("gi module:", gi.__file__)
    print("has rasterize_gaussian_higs_dynamic:", hasattr(gi, "rasterize_gaussian_higs_dynamic"))
    # Look for _C or _torch_impl style loader
    for name in dir(gi):
        if 'load' in name.lower() or name.startswith('_') and 'C' in name:
            print("  loader-ish attr:", name)
except Exception as e:
    import traceback; traceback.print_exc()
    print("GI_FAIL:", e)

# Check for scene_cuda extension availability
print("\n=== scene cuda extension ===")
try:
    import gsplat_scene_cuda
    print("gsplat_scene_cuda import: OK", gsplat_scene_cuda.__file__)
except Exception as e:
    print("gsplat_scene_cuda import (direct):", e)

# Check the render kernels/cuda build.py to understand the ext name
print("\n=== kernels/cuda build.py head ===")
try:
    from gsplat.experimental.render.kernels import cuda as kc
    print("kernels.cuda:", kc.__file__)
except Exception as e:
    print("kernels.cuda import:", e)

# List all loaded modules containing 'gsplat' or 'scene_cuda'
print("\n=== loaded modules ===")
for name, mod in sorted(sys.modules.items()):
    if name and ('gsplat' in name or 'scene_cuda' in name):
        f = getattr(mod, '__file__', None)
        print(f"  {name} -> {f}")
