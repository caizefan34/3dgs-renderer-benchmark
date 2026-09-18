#!/usr/bin/env python3
"""Build script for the rasterize_to_pixels_bwd_with_skip CUDA extension.

Compiles a custom CUDA kernel that adds a ``skip_mask`` parameter to gsplat's
``rasterize_to_pixels_bwd`` backward pass, enabling certificate-guided skipping
of gradient computation for certain Gaussian intersections while preserving
correctness of subsequent Gaussians' gradients.

The extension exposes a single function::

    rasterize_to_pixels_bwd_with_skip(means2d, conics, colors, opacities,
        backgrounds, masks, image_width, image_height, tile_size,
        tile_offsets, flatten_ids, render_alphas, last_ids,
        v_render_colors, v_render_alphas, skip_mask, absgrad)
      -> (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities)

Usage (on the remote machine via ``ssh mx``)::

    CUDA_HOME=/tmp/cuda_12_4 PYTHONNOUSERSITE=1 \\
        ~/miniforge3/envs/anysplat/bin/python experiments/r4/build_cuda_extension.py

Environment variables:
    CUDA_HOME            CUDA toolkit root (default: /tmp/cuda_12_4)
    TORCH_CUDA_ARCH_LIST GPU arch override (default: "8.0" for A100 sm_80)
    VERBOSE              Set to "1" for extra build output (always on here)
"""

import glob
import os
import shutil
import sys
import sysconfig


def _die(msg, code=1):
    print(f"\n{'=' * 60}")
    print(f"BUILD FAILED: {msg}")
    print(f"{'=' * 60}")
    sys.exit(code)


def _resolve_cuda_home():
    """Locate the CUDA toolkit, preferring the CUDA_HOME env var."""
    cuda_home = os.environ.get("CUDA_HOME", "")
    if cuda_home and os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")):
        return cuda_home

    # Fallback: /tmp/cuda_12_4 (specified in the task)
    fallback = "/tmp/cuda_12_4"
    if os.path.isfile(os.path.join(fallback, "bin", "nvcc")):
        # Check if it has cicc (complete installation)
        if not os.path.isfile(os.path.join(fallback, "nvvm", "bin", "cicc")):
            print(f"[build] WARNING: {fallback} has nvcc but is missing nvvm/bin/cicc")
            # Try higs-13scene-env (CUDA 12.8, complete installation)
            higs = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"
            if os.path.isfile(os.path.join(higs, "bin", "nvcc")) and \
               os.path.isfile(os.path.join(higs, "nvvm", "bin", "cicc")):
                print(f"[build] Using complete CUDA 12.8 at {higs}")
                return higs
        else:
            return fallback

    # Last resort: search PATH for nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))

    _die(
        f"nvcc not found. Tried CUDA_HOME={cuda_home!r}, "
        f"{fallback}/bin/nvcc, and PATH. Set CUDA_HOME to your CUDA toolkit root."
    )


def _find_gsplat_headers():
    """Locate gsplat's header directories.

    The installed gsplat pip package ships Common.h, Utils.cuh, Ops.h,
    Cameras.cuh in cuda/include/, but NOT bindings.h, helpers.cuh, or
    types.cuh (those are build-time-only headers). We fall back to the
    local canonical_source audit copy for those.
    """
    import gsplat

    pkg_dir = os.path.dirname(os.path.abspath(gsplat.__file__))
    cuda_dir = os.path.join(pkg_dir, "cuda")
    print(f"[build] gsplat {getattr(gsplat, '__version__', '?')} at {pkg_dir}")

    # Headers from the installed package
    gsplat_include = os.path.join(cuda_dir, "include")  # Common.h, Utils.cuh, Ops.h, Cameras.cuh
    gsplat_csrc = os.path.join(cuda_dir, "csrc")  # Rasterization.h, etc.
    glm_dir = os.path.join(gsplat_csrc, "third_party", "glm")  # glm/glm.hpp

    # Local canonical source (build-time-only headers not shipped in pip)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    canonical = os.path.join(repo_root, "candidate_c_source_audit", "canonical_source")
    canonical_raster_bwd = os.path.join(canonical, "raster_backward")

    # Verify required headers, checking installed package first, then canonical
    required = {
        "bindings.h": [
            os.path.join(gsplat_include, "bindings.h"),
            os.path.join(canonical, "bindings.h"),
        ],
        "helpers.cuh": [
            os.path.join(gsplat_csrc, "helpers.cuh"),
            os.path.join(canonical_raster_bwd, "helpers.cuh"),
        ],
        "types.cuh": [
            os.path.join(gsplat_csrc, "types.cuh"),
            os.path.join(canonical_raster_bwd, "types.cuh"),
        ],
        "glm/glm.hpp": [
            os.path.join(glm_dir, "glm", "glm.hpp"),
        ],
    }

    extra_include_paths = set()
    for label, candidates in required.items():
        found = False
        for path in candidates:
            if os.path.isfile(path):
                print(f"[build] Found {label} at {path}")
                extra_include_paths.add(os.path.dirname(path))
                found = True
                break
        if not found:
            _die(f"Required gsplat header not found: {label}. Searched: {candidates}")

    # Also add the parent of raster_backward so #include "raster_backward/helpers.cuh" works
    # and the canonical dir itself for #include "bindings.h"
    extra_include_paths.add(canonical)
    extra_include_paths.add(canonical_raster_bwd)
    # Also add installed package include paths for Common.h, Ops.h, etc.
    extra_include_paths.add(gsplat_include)
    extra_include_paths.add(gsplat_csrc)
    extra_include_paths.add(glm_dir)

    print(f"[build] All headers verified")

    return list(extra_include_paths)


def main():
    print("=" * 60)
    print("Building rasterize_to_pixels_bwd_with_skip CUDA extension")
    print("=" * 60)

    # --- 1. CUDA_HOME ---------------------------------------------------
    cuda_home = _resolve_cuda_home()
    os.environ["CUDA_HOME"] = cuda_home
    print(f"[build] CUDA_HOME  = {cuda_home}")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    # --- 2. Locate gsplat headers ---------------------------------------
    extra_include_paths = _find_gsplat_headers()

    # --- 3. Torch + Python include paths --------------------------------
    import torch

    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    python_include = sysconfig.get_path("include")
    print(f"[build] torch {torch.__version__} at {torch_dir}")
    print(f"[build] python include = {python_include}")

    if not torch.cuda.is_available():
        print("[build] WARNING: torch.cuda.is_available() is False — "
              "compilation may still succeed but the kernel cannot run here.")

    extra_include_paths.extend([
        torch_include,
        torch_api_include,
        python_include,
        # CUDA 12.x nv/target headers — /tmp/cuda_12_4 is incomplete (missing
        # nv/ directory), but the conda env has them. Add the conda include
        # root so #include <nv/target> resolves.
        os.path.dirname(python_include),  # .../envs/anysplat/include
    ])

    # --- 4. Source files ------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file = os.path.join(script_dir, "rasterize_to_pixels_bwd_with_skip.cu")
    cpp_file = os.path.join(script_dir, "ext_skip.cpp")

    for f in (cu_file, cpp_file):
        if not os.path.isfile(f):
            _die(f"Source file not found: {f}")
    print(f"[build] .cu source = {cu_file}")
    print(f"[build] .cpp source = {cpp_file}")

    sources = [cu_file, cpp_file]

    # --- 5. Compiler flags ----------------------------------------------
    # nvcc flags — match gsplat's own setup.py plus A100 arch targeting.
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        # Suppress GLM/Torch warning spam (same as gsplat setup.py)
        "-diag-suppress", "20012,186",
        "-diag-suppress", "177,240",
        # Safety net if the system gcc is newer than CUDA 12.4 officially supports
        "--allow-unsupported-compiler",
    ]

    # GPU architecture: use TORCH_CUDA_ARCH_LIST if set, otherwise default
    # to sm_80 (A100, the remote machine's GPU).
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list:
        print(f"[build] TORCH_CUDA_ARCH_LIST = {arch_list!r} (BuildExtension adds gencode)")
    else:
        print("[build] TORCH_CUDA_ARCH_LIST not set — defaulting to sm_80 (A100)")
        extra_cuda_cflags += [
            "-gencode=arch=compute_80,code=compute_80",  # PTX (forward-compat)
            "-gencode=arch=compute_80,code=sm_80",  # SASS (native A100)
        ]

    # C++ (host compiler) flags
    extra_cflags = [
        "-O3",
        "-std=c++17",
        "-Wno-sign-compare",
        "-Wno-attributes",
    ]

    # --- 6. Build directory ---------------------------------------------
    build_dir = os.path.join(script_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    # --- 7. Compile via torch.utils.cpp_extension.load() ----------------
    print("\n[build] Starting JIT compilation...")
    print(f"[build]   include paths: {extra_include_paths}")
    print(f"[build]   nvcc flags:    {' '.join(extra_cuda_cflags)}")
    print(f"[build]   cxx flags:     {' '.join(extra_cflags)}")
    print(f"[build]   build dir:     {build_dir}")

    from torch.utils.cpp_extension import load

    try:
        ext = load(
            name="ext_skip",
            sources=sources,
            extra_include_paths=extra_include_paths,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=build_dir,
            verbose=True,
        )
    except Exception as exc:
        _die(str(exc))

    # --- 8. Report results ----------------------------------------------
    so_files = sorted(glob.glob(os.path.join(build_dir, "**", "*.so"), recursive=True))
    # Also check the top-level build dir (load() may place it there)
    so_files += sorted(glob.glob(os.path.join(build_dir, "*.so")))

    print(f"\n{'=' * 60}")
    print("BUILD SUCCESSFUL")
    print(f"{'=' * 60}")

    if so_files:
        seen = set()
        for so in so_files:
            if so in seen:
                continue
            seen.add(so)
            size_kb = os.path.getsize(so) / 1024.0
            print(f"  .so path: {so}")
            print(f"  .so size: {size_kb:.1f} KB")
    else:
        print("  WARNING: no .so file found in build directory")

    # Verify the module exposes the expected function
    func_name = "rasterize_to_pixels_bwd_with_skip"
    if hasattr(ext, func_name):
        print(f"  Module function: {func_name} ✓")
    else:
        avail = [a for a in dir(ext) if not a.startswith("_")]
        print(f"  WARNING: module does not expose '{func_name}'")
        print(f"  Available: {avail}")

    ext_file = getattr(ext, "__file__", None)
    print(f"\n  Extension module: {ext}")
    if ext_file:
        print(f"  Module file:      {ext_file}")

    print("\nDone. To use the extension:")
    print(f'  import sys; sys.path.insert(0, r"{build_dir}")')
    print(f'  from ext_skip import rasterize_to_pixels_bwd_with_skip')
    print("  # or: torch.ops.load_library(r'<path-to-.so>')")


if __name__ == "__main__":
    main()
