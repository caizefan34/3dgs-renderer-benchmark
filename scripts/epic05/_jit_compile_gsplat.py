"""Trigger gsplat 1.5.3 JIT compilation from pip source."""
import os
import sys

# Ensure CUDA_PATH has no trailing space
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3".strip()
os.environ["CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DISTUTILS_USE_SDK"] = "1"
os.environ["MAX_JOBS"] = "2"

# Fix CJK locale issue for subprocess
import torch.utils.cpp_extension as cpp_ext
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')

# Add MSVC and CUDA to PATH for subprocess
_msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_cuda_bin = os.environ["CUDA_PATH"] + r"\bin"
_extra_paths = [_msvc_dir, _cuda_bin]
current_path = os.environ.get("PATH", "")
for p in _extra_paths:
    if p not in current_path:
        current_path = p + os.pathsep + current_path
os.environ["PATH"] = current_path

print(f"CUDA_PATH = {os.environ['CUDA_PATH']!r}")
print(f"CUDA bin = {_cuda_bin!r}")
print(f"MSVC dir = {_msvc_dir!r}")
print(f"cl.exe exists: {os.path.exists(os.path.join(_msvc_dir, 'cl.exe'))}")

# Force rebuild gsplat CUDA extension
import gsplat.cuda._backend as bk
print("Calling load_extension(force_rebuild=True)...")
try:
    bk.load_extension(force_rebuild=True)
    print("JIT load_extension completed")
except Exception as e:
    print(f"JIT failed: {type(e).__name__}: {e}")
    sys.exit(1)

# Verify
from gsplat import csrc
print(f"csrc module: {csrc}")
import gsplat.cuda._C as C
print(f"_C module: {C}")

# Test all tile sizes
import torch
device = 'cuda'
means = torch.randn(100, 3, device=device)
quats = torch.nn.functional.normalize(torch.randn(100, 4, device=device), dim=-1)
scales = torch.rand(100, 3, device=device) * 0.1
opacities = torch.sigmoid(torch.randn(100, device=device))
colors = torch.randn(100, 16, 3, device=device)
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[500, 0, 960], [0, 500, 540], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

from gsplat import rasterization
for ts in [8, 16, 32]:
    try:
        rendered, alpha, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmat, Ks=K, width=1920, height=1080,
            tile_size=ts, sh_degree=3, packed=True, render_mode='RGB'
        )
        print(f"  tile_size={ts}: OK, shape={rendered.shape}")
    except Exception as e:
        err = str(e).strip()[:120]
        print(f"  tile_size={ts}: FAILED - {err}")

print("ALL DONE")
