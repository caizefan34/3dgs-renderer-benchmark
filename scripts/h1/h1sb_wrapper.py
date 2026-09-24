#!/usr/bin/env python3
"""Wrapper to load AccuTile gsplat with pre-compiled extension on torch 2.9.1+cu128."""
import os, sys, shutil

# 1. Patch the versioner to handle list arguments (torch 2.9.1 bug)
import torch.utils._cpp_extension_versioner as _ver
_orig_update_hash = _ver.update_hash
def _patched_update_hash(seed, value):
    if isinstance(value, list):
        value = tuple(value)
    return _orig_update_hash(seed, value)
_ver.update_hash = _patched_update_hash

# 2. Copy pre-compiled extension from cu124 to cu128 if not present
src_dir = os.path.expanduser("~/.cache/torch_extensions/py310_cu124/gsplat_cuda")
dst_dir = os.path.expanduser("~/.cache/torch_extensions/py310_cu128/gsplat_cuda")
if os.path.exists(src_dir) and not os.path.exists(os.path.join(dst_dir, "gsplat_cuda.so")):
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        src_f = os.path.join(src_dir, f)
        dst_f = os.path.join(dst_dir, f)
        if os.path.isfile(src_f) and not os.path.exists(dst_f):
            try:
                shutil.copy2(src_f, dst_f)
            except Exception:
                pass
    print("Copied pre-compiled extension from cu124 to cu128")

# 3. Now run the actual profiling script
ACCUTILE_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
sys.path.insert(0, ACCUTILE_TREE)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

# Test import
import gsplat
print("gsplat file:", gsplat.__file__)
print("gsplat version:", gsplat.__version__)

# Check if accutile param exists
import inspect
from gsplat.rendering import rasterization
sig = inspect.signature(rasterization)
if "accutile" in sig.parameters:
    print("accutile parameter: AVAILABLE")
else:
    print("accutile parameter: NOT FOUND")

# Run the profiling
exec(open("/tmp/h1sb_b1a_profile.py").read())