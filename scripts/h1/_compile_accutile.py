import os, sys, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["PYTHONNOUSERSITE"] = "1"
CONDA_PREFIX = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"
os.environ["CUDA_HOME"] = CONDA_PREFIX
# Add CUDA headers to include path (conda puts them in targets/x86_64-linux/include)
cuda_include = os.path.join(CONDA_PREFIX, "targets", "x86_64-linux", "include")
import torch
from torch.utils.cpp_extension import load
PATH = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153/gsplat/cuda"
glm = os.path.join(PATH, "csrc", "third_party", "glm")
sources = sorted(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + sorted(glob.glob(os.path.join(PATH, "csrc/*.cpp"))) + [os.path.join(PATH, "ext.cpp")]
print("Sources:", len(sources))
print("CUDA include:", cuda_include)
try:
    _C = load(name="gsplat_cuda", sources=sources, extra_cflags=["-O3", "-Wno-attributes"], extra_cuda_cflags=["-O3", "-use_fast_math", "--extended-lambda"], extra_include_paths=[os.path.join(PATH, "include/"), glm, cuda_include], verbose=True)
    print("SUCCESS")
except Exception as e:
    print("FAILED:", e)
    # Check the build log
    build_dir = os.path.expanduser("~/.cache/torch_extensions/py310_cu128/gsplat_cuda")
    log = os.path.join(build_dir, "build.ninja")
    if os.path.exists(log):
        print("Build dir:", build_dir)