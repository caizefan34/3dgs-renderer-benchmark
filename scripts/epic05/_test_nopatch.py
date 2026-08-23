#!/usr/bin/env python3
"""Test if gsplat works WITHOUT any patch, just with MSVC in PATH."""
import os, sys

# Set MSVC and CUDA in PATH at the very start
msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
extra_paths = [msvc_dir, cuda_bin]

for p in extra_paths:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Verify cl.exe is findable via os.environ PATH
import subprocess
try:
    result = subprocess.run(["where", "cl"], capture_output=True, text=True, check=True)
    print(f"cl.exe found: {result.stdout.strip()}", flush=True)
except:
    print("WARNING: cl.exe not found in PATH", flush=True)

import torch
import torch.utils.cpp_extension as cpp_ext
import functools, inspect

# ALSO need to patch cpp_ext.get_compiler_abi_compatibility to skip the cl check
# Actually, the cl check is just a warning in _write_ninja_file
# The real issue might be something else

# Patch: direct fix - check if the retry call needs cl.exe
# Let's just try importing and see what happens
print("Importing gsplat rasterization...", flush=True)
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
