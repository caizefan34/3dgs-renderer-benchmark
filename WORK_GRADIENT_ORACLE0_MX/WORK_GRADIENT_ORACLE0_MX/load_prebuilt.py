#!/usr/bin/env python3
"""Load prebuilt gsplat_scene_cuda from build dir, then run Gate D/E/F."""
import os, sys, glob

BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
print("build dir exists:", os.path.isdir(BUILD_DIR))
print(".so files:", glob.glob(BUILD_DIR + "/*.so"))
# Prepend so prebuilt import works (no JIT rebuild)
sys.path.insert(0, BUILD_DIR)
# Redirect any JIT build attempts to storage pool
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"

import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)

# Try prebuilt import directly
try:
    import gsplat_scene_cuda
    print("gsplat_scene_cuda prebuilt import: OK")
    print("attrs:", [a for a in dir(gsplat_scene_cuda) if not a.startswith('__')][:40])
except Exception as e:
    print("prebuilt import FAILED:", e)
    gsplat_scene_cuda = None
