#!/usr/bin/env python3
"""R2.1: Build v4 — gcc-11 for .cpp, gcc-10 for .cu, version script for linking."""
import os, sys, subprocess, glob

GSPLAT_ROOT = "/tmp/gsplat_baseline/gsplat-1.5.3"
BUILD_TEMP = f"{GSPLAT_ROOT}/build/temp.linux-x86_64-3.10/gsplat/cuda/csrc"
BUILD_CUDA = f"{GSPLAT_ROOT}/build/temp.linux-x86_64-3.10/gsplat/cuda"

import torch
TORCH_INC = os.path.join(os.path.dirname(torch.__file__), "include")
TORCH_API_INC = os.path.join(TORCH_INC, "torch/csrc/api/include")

INCLUDE_DIRS = [
    f"{GSPLAT_ROOT}/gsplat/cuda/csrc/third_party/glm",
    f"{GSPLAT_ROOT}/gsplat/cuda/include",
    TORCH_INC,
    TORCH_API_INC,
    "/usr/include/python3.10",
]
INC_FLAGS = " ".join([f"-I{d}" for d in INCLUDE_DIRS])

NVCC_FLAGS = (
    f"--allow-unsupported-compiler "
    f"-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ "
    f"-D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ "
    f"--expt-relaxed-constexpr --compiler-options -fPIC -O3 --use_fast_math "
    f"-std=c++17 --expt-relaxed-constexpr -diag-suppress 20012,186 "
    f"-DTORCH_API_INCLUDE_EXTENSION_H "
    f"-DPYBIND11_COMPILER_TYPE=\\\"_gcc\\\" "
    f"-DPYBIND11_STDLIB=\\\"_libstdcpp\\\" "
    f"-DPYBIND11_BUILD_ABI=\\\"_cxxabi1016\\\" "
    f"-DTORCH_EXTENSION_NAME=csrc -D_GLIBCXX_USE_CXX11_ABI=1 "
    f"-gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 "
    f"-diag-suppress 177,240"
)

GCC_FLAGS = (
    f"-DNDEBUG -O2 -fPIC "
    f"-O3 -Wno-sign-compare -DAT_PARALLEL_OPENMP -fopenmp "
    f"-DTORCH_API_INCLUDE_EXTENSION_H "
    f"-DPYBIND11_COMPILER_TYPE=\\\"_gcc\\\" "
    f"-DPYBIND11_STDLIB=\\\"_libstdcpp\\\" "
    f"-DPYBIND11_BUILD_ABI=\\\"_cxxabi1016\\\" "
    f"-DTORCH_EXTENSION_NAME=csrc -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17 "
    f"-Wno-attributes"
)

os.chdir(GSPLAT_ROOT)
os.makedirs(BUILD_TEMP, exist_ok=True)

# Use original gcc-10 wrappers (with gcc-11 include-fixed) for nvcc
GCC10_WRAP = "/tmp/gcc10_wrappers/gcc"

# Step 1: Compile .cu with nvcc + gcc-10
print("=== Compiling RasterizeToPixels3DGSBwd.cu (nvcc + gcc-10) ===")
cu_file = "gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu"
o_file = f"{BUILD_TEMP}/RasterizeToPixels3DGSBwd.o"
cmd = f"/usr/bin/nvcc -ccbin {GCC10_WRAP} {INC_FLAGS} -c {cu_file} -o {o_file} {NVCC_FLAGS}"
ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if ret.returncode != 0:
    print(f"  FAILED:\n{ret.stderr[-2000:]}")
    sys.exit(1)
print("  OK")

# Step 2: Compile .cpp with system g++-11
print("\n=== Compiling Rasterization.cpp (system g++-11) ===")
cpp_file = "gsplat/cuda/csrc/Rasterization.cpp"
o_file2 = f"{BUILD_TEMP}/Rasterization.o"
cmd2 = f"x86_64-linux-gnu-g++ {INC_FLAGS} -c {cpp_file} -o {o_file2} {GCC_FLAGS}"
ret2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
if ret2.returncode != 0:
    print(f"  FAILED:\n{ret2.stderr[:1500]}")
    sys.exit(1)
print("  OK")

# Step 2b: Compile ext.cpp with system g++-11
print("\n=== Compiling ext.cpp (system g++-11) ===")
ext_cpp = "gsplat/cuda/ext.cpp"
ext_o = f"{BUILD_CUDA}/ext.o"
cmd_ext = f"x86_64-linux-gnu-g++ {INC_FLAGS} -c {ext_cpp} -o {ext_o} {GCC_FLAGS}"
ret_ext = subprocess.run(cmd_ext, shell=True, capture_output=True, text=True)
if ret_ext.returncode != 0:
    print(f"  FAILED:\n{ret_ext.stderr[:1500]}")
    sys.exit(1)
print("  OK")

# Step 3: Re-link with version script to suppress symbol versioning
print("\n=== Re-linking csrc.so (with version script) ===")
all_o_files = sorted(set(
    glob.glob(f"{BUILD_TEMP}/*.o") +
    glob.glob(f"{BUILD_CUDA}/ext.o")
))
print(f"  {len(all_o_files)} .o files")

# Create version map
version_map = "/tmp/r21_version.map"
with open(version_map, "w") as f:
    f.write("/* version map */\n{\nglobal: *;\n};\n")

so_output = f"{GSPLAT_ROOT}/gsplat/csrc.so"
link_flags = f"-shared -fopenmp -O2 -Wl,--version-script={version_map}"
link_cmd = f"x86_64-linux-gnu-g++ {' '.join(all_o_files)} -o {so_output} {link_flags}"
ret3 = subprocess.run(link_cmd, shell=True, capture_output=True, text=True)
if ret3.returncode != 0:
    print(f"  FAILED:\n{ret3.stderr[-1000:]}")
    sys.exit(1)
print("  OK")
print(f"  .so size: {os.path.getsize(so_output)} bytes")

# Strip
subprocess.run(f"strip {so_output}", shell=True)
print(f"  Stripped: {os.path.getsize(so_output)} bytes")

# Step 4: Verify
print("\n=== Verifying ===")
import gsplat
print(f"  gsplat: {gsplat.__file__}")
from gsplat.cuda._backend import _C
print(f"  _C loaded: {_C}")
print(f"  rasterize_to_pixels_3dgs_bwd: {hasattr(_C, 'rasterize_to_pixels_3dgs_bwd')}")
print("\n=== Build successful ===")
