#!/usr/bin/env python3
"""FINAL gate check: gsplat import + camera seq + R3 runner dependencies."""
import os, sys, json, glob, traceback
import torch

print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print()

# 1. gsplat import test
try:
    import gsplat
    print(f"gsplat import: OK (version {getattr(gsplat, '__version__', 'unknown')})")
    
    # try loading package submodules we need
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels
    )
    print("gsplat.cuda._wrapper: OK")
except Exception as e:
    print(f"gsplat import FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# 2. camera sequence
for p in ["data/camera_sequence.npy", "data/camera_sequence_30000.npy", "data/camera_seq.npy"]:
    if os.path.exists(p):
        import numpy as np
        arr = np.load(p)
        print(f"\ncamera seq: {p} shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")

# 3. check R3 script's camera handling
sys.path.insert(0, "experiments/r3")
try:
    import r3_certificate_runner as r
    src = open("experiments/r3/r3_certificate_runner.py").read()
    import re
    # find how camera seq is used
    for m in re.finditer(r'camera_sequence', src):
        line_no = src[:m.start()].count('\n') + 1
        line = src.split('\n')[line_no-1].strip()
        if 'def ' not in line:
            print(f"  L{line_no}: {line[:100]}")
except Exception as e:
    print(f"R3 runner import issue: {type(e).__name__}: {e}")
