"""Test loading rebuilt .so with RTLD_GLOBAL flag."""
import os, sys

# Save original flags
old_flags = sys.getdlopenflags()
# Set RTLD_GLOBAL so PyTorch's symbols are available to the .so
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

import torch
print("torch loaded")

# Now try loading gsplat
try:
    from gsplat.cuda._backend import _C
    print(f"_C loaded: {_C}")
    print(f"rasterize_to_pixels_3dgs_bwd: {hasattr(_C, 'rasterize_to_pixels_3dgs_bwd')}")
except Exception as e:
    print(f"_C import FAILED: {e}")

# Restore flags
sys.setdlopenflags(old_flags)
