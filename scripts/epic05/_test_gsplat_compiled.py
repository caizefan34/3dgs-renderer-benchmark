"""Test the freshly compiled gsplat CUDA extension with all tile sizes."""
import os
import sys

# Ensure cl.exe is on PATH before importing anything
_msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
for p in [_msvc_dir, _cuda_bin]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3".strip()

# Fix CJK locale issue for subprocess calls
import torch.utils.cpp_extension as cpp_ext
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')

print("Importing gsplat...")
import gsplat
print(f"gsplat version: {gsplat.__version__}")

# _C should be loaded by _backend.py
from gsplat.cuda._backend import _C
print(f"_C module: {_C}")

# Check which ops are available
_c = _C
has_3dgs_fwd = hasattr(_c, 'rasterize_to_pixels_3dgs_fwd')
has_2dgs_fwd = hasattr(_c, 'rasterize_to_pixels_2dgs_fwd')
has_from_world_fwd = hasattr(_c, 'rasterize_to_pixels_from_world_3dgs_fwd')
print(f"  3DGS fwd: {has_3dgs_fwd}")
print(f"  2DGS fwd: {has_2dgs_fwd}")
print(f"  FromWorld fwd: {has_from_world_fwd}")

# Test 3DGS rasterization with all tile sizes
import torch
print("\nTesting tile sizes...")
device = 'cuda'
means = torch.randn(100, 3, device=device)
quats = torch.nn.functional.normalize(torch.randn(100, 4, device=device), dim=-1)
scales = torch.rand(100, 3, device=device) * 0.1
opacities = torch.sigmoid(torch.randn(100, device=device))
colors = torch.randn(100, 16, 3, device=device)
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[500, 0, 960], [0, 500, 540], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

from gsplat import rasterization
for ts in [4, 8, 16, 32]:
    try:
        rendered, alpha, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmat, Ks=K, width=1920, height=1080,
            tile_size=ts, sh_degree=3, packed=True, render_mode='RGB'
        )
        print(f"  tile_size={ts}: OK, shape={rendered.shape}")
    except Exception as e:
        import traceback
        err = traceback.format_exc()[:200]
        print(f"  tile_size={ts}: FAILED")
        print(f"    {err.split(chr(10))[0]}")

print("\nALL DONE")
