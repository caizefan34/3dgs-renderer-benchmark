#!/usr/bin/env python3
"""Wrapper to load AccuTile gsplat with pre-compiled .so, bypassing JIT compile."""
import os, sys, types, importlib.util, ctypes

# 1. Add torch lib to LD_LIBRARY_PATH and preload libc10
import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# Preload libc10.so globally so the extension can find it
ctypes.CDLL(os.path.join(torch_lib, "libc10.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(torch_lib, "libc10_cuda.so"), mode=ctypes.RTLD_GLOBAL)
print("Preloaded torch libs from:", torch_lib)

# 2. Load the pre-compiled .so as a Python module
so_path = os.path.expanduser("~/.cache/torch_extensions/py310_cu124/gsplat_cuda/gsplat_cuda.so")
print("Loading pre-compiled extension:", so_path)

spec = importlib.util.spec_from_file_location("gsplat_cuda", so_path)
gsplat_cuda_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gsplat_cuda_mod)
print("Extension loaded successfully")

# 3. Create a fake _backend module
fake_backend = types.ModuleType("gsplat.cuda._backend")
fake_backend._C = gsplat_cuda_mod
sys.modules["gsplat.cuda._backend"] = fake_backend

# 4. Import gsplat from AccuTile tree
ACCUTILE_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
sys.path.insert(0, ACCUTILE_TREE)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

import gsplat
print("gsplat file:", gsplat.__file__)
from gsplat.rendering import rasterization
import inspect
sig = inspect.signature(rasterization)
print("accutile param:", "accutile" in sig.parameters)

# 5. Quick smoke test
import numpy as np
N = 100; device = "cuda:0"
means = torch.randn(N, 3, device=device)
quats = torch.tensor([[1,0,0,0]], dtype=torch.float32, device=device).repeat(N, 1)
scales = torch.ones(N, 3, device=device) * 0.1
opacities = torch.ones(N, device=device)
colors = torch.zeros(1, N, 16, 3, device=device); colors[:, :, 0] = 0.5
vm = torch.eye(4, device=device).reshape(1, 1, 4, 4)
K = torch.tensor([[100,0,50],[0,100,50],[0,0,1]], device=device, dtype=torch.float32).reshape(1,1,3,3)
with torch.no_grad():
    out = rasterization(means=means.unsqueeze(0), quats=quats.unsqueeze(0), scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0), colors=colors, viewmats=vm, Ks=K, width=100, height=100,
        sh_degree=3, packed=True, accutile=True)
print("Smoke test PASSED! Render shape:", tuple(out[0].shape))

# 6. Run the profiling script
print("\nStarting B1A profiling...")
exec(open("/tmp/h1sb_b1a_profile.py").read())