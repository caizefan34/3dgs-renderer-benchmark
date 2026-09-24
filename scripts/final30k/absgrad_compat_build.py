#!/usr/bin/env python3
"""Build C0_V3_FINAL30K composed .so (absgrad compatibility artifact).

Replicates the frozen C0 build (higs_c0_cache_composed) exactly — same
sources, include paths, cflags, cuda_cflags, ldflags — against the patched
worktree higs_c0_final30k_worktree, into a NEW cache dir. The old artifact
(higs_c0_cache_composed, sha 7ca1c6bf...) is never touched.
"""
import hashlib
import json
import os
import subprocess
import sys

import torch
from torch.utils import cpp_extension

WT = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
BUILD_DIR = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_cache/experimental_gaussian_render_inference_scene_cuda"
NAME = "experimental_gaussian_render_inference_scene_cuda"

os.makedirs(BUILD_DIR, exist_ok=True)

sources = [
    f"{WT}/gsplat/experimental/render/kernels/cuda/ext.cpp",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/GaussianRenderInferenceScene.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/GatherVisible.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/GatherlessProjectedProducer.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/IntersectCommon.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/IntersectMTFused.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/MacroTileIntersect.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/MacroTileRasterize.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/Projection.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/SegmentedSort.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/SHCompression.cu",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/SphericalHarmonics.cu",
]
extra_include_paths = [
    f"{WT}/gsplat/cuda/include",
    f"{WT}/gsplat/cuda/csrc",
    f"{WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference",
    f"{WT}/gsplat/cuda/csrc/third_party/glm",
    "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/targets/x86_64-linux/include",
]
extra_cflags = ["-std=c++20", "-O3", "-DNDEBUG", "-Wno-attributes", "-Wno-unknown-pragmas", "-DAT_PARALLEL_OPENMP", "-fopenmp"]
extra_cuda_cflags = ["--forward-unknown-opts", "-use_fast_math", "-diag-suppress", "20012,186", "--expt-relaxed-constexpr", "-std=c++20", "-O3", "-DNDEBUG", "-Wno-attributes", "-Wno-unknown-pragmas", "-DAT_PARALLEL_OPENMP", "-fopenmp"]
extra_ldflags = ["-s"]

print(f"torch: {torch.__version__}  cuda avail: {torch.cuda.is_available()}")
print(f"nvcc: {cpp_extension.CUDA_HOME}")

mod = cpp_extension.load(
    name=NAME,
    sources=sources,
    extra_include_paths=extra_include_paths,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    extra_ldflags=extra_ldflags,
    build_directory=BUILD_DIR,
    verbose=True,
)
print("MODULE IMPORT OK:", mod)

so = os.path.join(BUILD_DIR, f"{NAME}.so")
sha = hashlib.sha256(open(so, "rb").read()).hexdigest()
print("SO_SHA256:", sha)

# build identity record
rec = {
    "artifact": "C0_V3_FINAL30K",
    "worktree": WT,
    "build_dir": BUILD_DIR,
    "so_path": so,
    "so_sha256": sha,
    "torch": torch.__version__,
    "cuda_home": cpp_extension.CUDA_HOME,
    "nvcc_version": subprocess.run(
        [os.path.join(cpp_extension.CUDA_HOME, "bin", "nvcc"), "--version"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()[-1],
    "gpu_arch_detected": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
    "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS")},
    "has_higs_rasterize_backward": hasattr(mod, "higs_rasterize_backward"),
}
with open("/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_build_identity.json", "w") as f:
    json.dump(rec, f, indent=2)
print("BUILD IDENTITY:", json.dumps(rec, indent=2))
