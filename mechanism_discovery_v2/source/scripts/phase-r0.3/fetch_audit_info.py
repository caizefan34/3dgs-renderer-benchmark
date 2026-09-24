import sys, os, json

# Check configs
for root, dirs, files in os.walk("configs"):
    for f in files:
        if f.endswith((".yaml", ".yml")):
            print(f"CONFIG: {os.path.join(root, f)}")

# Check tests
for root, dirs, files in os.walk("tests"):
    for f in files:
        if f.endswith(".py"):
            print(f"TEST: {os.path.join(root, f)}")

# Check gsplat
import gsplat
print(f"GSPLAT_FILE: {gsplat.__file__}")
print(f"GSPLAT_VER: {gsplat.__version__}")

# Check CUDA source files in gsplat
gdir = os.path.dirname(gsplat.__file__)
cuda_dir = os.path.join(gdir, "cuda")
if os.path.isdir(cuda_dir):
    print(f"GSPLAT_CUDA_DIR: {cuda_dir}")
    for f in sorted(os.listdir(cuda_dir)):
        print(f"GSPLAT_CUDA: {f}")

# Check patched source
for p in ["/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda"]:
    if os.path.isdir(p):
        print(f"PATCHED_CUDA_DIR: {p}")
        for f in sorted(os.listdir(p)):
            print(f"PATCHED_CUDA: {f}")

# PyTorch info
import torch
print(f"TORCH_VER: {torch.__version__}")
print(f"CUDA_VER: {torch.version.cuda}")
print(f"CUDA_ARCH: {torch.cuda.get_arch_list()}")
