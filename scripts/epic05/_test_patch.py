#!/usr/bin/env python3
"""Test if the _jit_compile patch makes gsplat rasterization work."""
import sys, os

# Add MSVC and CUDA to PATH
msvc = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
os.environ["PATH"] = msvc + os.pathsep + cuda + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

import torch
import torch.utils.cpp_extension as cpp_ext
import functools, inspect

# Patch _jit_compile 
_orig = cpp_ext._jit_compile
_sig = inspect.signature(_orig)
if "extra_sycl_cflags" in _sig.parameters:
    @functools.wraps(_orig)
    def _patched(name, sources, extra_cflags, extra_cuda_cflags,
                 extra_sycl_cflags=None, extra_ldflags=None,
                 extra_include_paths=None, build_directory=None,
                 verbose=False, with_cuda=None, with_sycl=False,
                 is_python_module=False, is_standalone=False,
                 keep_intermediates=False):
        # Handle old-style calls that pass extra_ldflags positionally as 5th arg
        # Old sig: (name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, ...)
        # New sig: (name, sources, extra_cflags, extra_cuda_cflags, extra_sycl_cflags, extra_ldflags, ...)
        # If called with old-style, shift: what was extra_ldflags -> extra_sycl_cflags, etc.
        if extra_sycl_cflags is not None and extra_ldflags is None:
            # Old-style call: the 5th positional was meant as extra_ldflags
            # Shift everything
            return _orig(
                name, sources, extra_cflags, extra_cuda_cflags,
                [], # extra_sycl_cflags (empty)
                extra_sycl_cflags,  # this was actually extra_ldflags
                extra_include_paths, build_directory, verbose,
                with_cuda if with_cuda is not None else None,
                False if not with_sycl else with_sycl,  # with_sycl
                is_python_module, is_standalone, keep_intermediates
            )
        return _orig(name, sources, extra_cflags, extra_cuda_cflags,
                     extra_sycl_cflags or [], extra_ldflags,
                     extra_include_paths, build_directory, verbose,
                     with_cuda, bool(with_sycl),
                     bool(is_python_module), bool(is_standalone),
                     bool(keep_intermediates))
    cpp_ext._jit_compile = _patched
    print("Patch applied", flush=True)
else:
    print("Patch not needed", flush=True)

# Try preloading cached extension
cache_dir = os.path.join(
    os.environ["LOCALAPPDATA"],
    "torch_extensions", "torch_extensions", "Cache", "py310_cu130", "gsplat_cuda"
)
if os.path.isdir(cache_dir):
    sys.path.insert(0, cache_dir)
    try:
        ext = __import__("gsplat_cuda")
        sys.modules["gsplat.csrc"] = ext
        print(f"Preloaded extension from cache: {cache_dir}", flush=True)
    except ImportError as e:
        print(f"Cache preload failed: {e}", flush=True)

from gsplat import rasterization
print("gsplat.rasterization imported OK", flush=True)

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
print(f"Rendered shape: {rendered.shape}", flush=True)
print("SUCCESS", flush=True)
