#!/usr/bin/env python3
"""Test gsplat with code page fix and proper PATH."""
import os, sys, subprocess

# Set MSVC and CUDA in PATH at the very start
msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"

for p in [msvc_dir, cuda_bin]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check cl.exe output
result = subprocess.run(["where", "cl"], capture_output=True, text=True)
print(f"cl.exe: {result.stdout.strip()}", flush=True)

# Check cl.exe version output encoding
result = subprocess.run(["cl"], capture_output=True, text=False)
raw = result.stdout + result.stderr
print(f"cl.exe raw bytes: {raw[:200]}", flush=True)

# Python's oem codec issue: chcp to English (437) doesn't help because
# PyTorch uses SUBPROCESS_DECODE_ARGS which is ('oem',) on Windows
# Let's monkey-patch that too
import torch.utils.cpp_extension as cpp_ext
import functools, inspect

# Patch SUBPROCESS_DECODE_ARGS to use 'utf-8' instead of 'oem'
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')

# Also need to patch _jit_compile signature (same as before)
_orig = cpp_ext._jit_compile
_sig = inspect.signature(_orig)

@functools.wraps(_orig)
def _patched(*a, **kw):
    if "extra_sycl_cflags" not in kw:
        kw["extra_sycl_cflags"] = []
    if "with_sycl" not in kw:
        kw["with_sycl"] = False
    return _orig(*a, **kw)
cpp_ext._jit_compile = _patched

print("Patches applied", flush=True)

import torch
from gsplat import rasterization
print("Import OK", flush=True)

device = "cuda"
means = torch.randn(100, 3, device=device)
quats = torch.nn.functional.normalize(torch.randn(100, 4, device=device), dim=-1)
scales = torch.rand(100, 3, device=device) * 0.1
opacities = torch.sigmoid(torch.randn(100, device=device))
colors = torch.randn(100, 16, 3, device=device)
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[500, 0, 960], [0, 500, 540], [0, 0, 1]], device=device).unsqueeze(0)

rendered, alpha, info = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities,
    colors=colors, viewmats=viewmat, Ks=K, width=1920, height=1080,
    tile_size=16, sh_degree=3, packed=True, render_mode="RGB"
)
print(f"Rendered: {rendered.shape}", flush=True)
print("SUCCESS", flush=True)
