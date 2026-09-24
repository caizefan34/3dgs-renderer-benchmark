#!/usr/bin/env python3
"""Get full signature of higs_rasterize_backward."""
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PATH"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"

import torch
import importlib.util

SOURCE = "/tmp/higs_h3_fwd_1a_source"
CORE_SO = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
EXP_SO = "/tmp/higs_h3_fwd_1a_build/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"

sys.path.insert(0, SOURCE)
spec0 = importlib.util.spec_from_file_location("gsplat_cuda", CORE_SO)
core = importlib.util.module_from_spec(spec0)
spec0.loader.exec_module(core)
sys.modules["gsplat.csrc"] = core

spec = importlib.util.spec_from_file_location("experimental_gaussian_render_inference_scene_cuda", EXP_SO)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

fn = m.higs_rasterize_backward
print("Full docstring:")
print(fn.__doc__)
print("\n\nType:", type(fn))

# Also check the _C module
from gsplat.cuda import _C
print("\n\n_C exports:", [x for x in dir(_C) if not x.startswith('_')])
bwd_c = [x for x in dir(_C) if 'backward' in x.lower() or 'bwd' in x.lower()]
print("_C backward:", bwd_c)
for name in bwd_c:
    obj = getattr(_C, name)
    if hasattr(obj, '__doc__') and obj.__doc__:
        print(f"\n{name}: {obj.__doc__[:500]}")
