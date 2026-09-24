#!/usr/bin/env python3
"""H1-SB B1A wrapper with compiled AccuTile extension."""
import os, sys, glob, types, importlib.util

CONDA_PREFIX = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"
os.environ["CUDA_HOME"] = CONDA_PREFIX
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"
cuda_include = os.path.join(CONDA_PREFIX, "targets", "x86_64-linux", "include")

# 1. Compile if needed, or load existing
import torch
from torch.utils.cpp_extension import load, _get_build_directory

build_dir = _get_build_directory("gsplat_cuda", verbose=False)
so_path = os.path.join(build_dir, "gsplat_cuda.so")

if os.path.exists(so_path):
    # Load existing
    import ctypes
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    ctypes.CDLL(os.path.join(torch_lib, "libc10.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(torch_lib, "libc10_cuda.so"), mode=ctypes.RTLD_GLOBAL)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", so_path)
    _C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_C)
    print("Loaded existing extension from", so_path)
else:
    # Compile
    ACCUTILE_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
    PATH = os.path.join(ACCUTILE_TREE, "gsplat", "cuda")
    glm = os.path.join(PATH, "csrc", "third_party", "glm")
    sources = sorted(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + sorted(glob.glob(os.path.join(PATH, "csrc/*.cpp"))) + [os.path.join(PATH, "ext.cpp")]
    _C = load(name="gsplat_cuda", sources=sources, extra_cflags=["-O3", "-Wno-attributes"],
              extra_cuda_cflags=["-O3", "-use_fast_math", "--extended-lambda"],
              extra_include_paths=[os.path.join(PATH, "include/"), glm, cuda_include], verbose=False)
    print("Compiled new extension")

# 2. Fake _backend
fake_backend = types.ModuleType("gsplat.cuda._backend")
fake_backend._C = _C
sys.modules["gsplat.cuda._backend"] = fake_backend

# 3. Import gsplat from AccuTile tree
ACCUTILE_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
sys.path.insert(0, ACCUTILE_TREE)
import gsplat
print("gsplat:", gsplat.__file__)

from gsplat.rendering import rasterization
import inspect
sig = inspect.signature(rasterization)
print("accutile param:", "accutile" in sig.parameters)

# 4. Smoke test
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
print("Smoke test PASSED! Shape:", tuple(out[0].shape))

# 5. Run profiling
print("\nStarting B1A profiling...")
exec(open("/tmp/h1sb_b1a_profile.py").read())