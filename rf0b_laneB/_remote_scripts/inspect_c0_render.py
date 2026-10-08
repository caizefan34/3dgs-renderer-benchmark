import inspect, sys, os
# C0 env
os.environ["HIGS_PX_RUNTIME"]="2"; os.environ["HIGS_BWD_SCALAR_ADJOINT"]="scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"]="1"; os.environ["HIGS_BWD_ABSGRAD"]="1"; os.environ.pop("HIGS_DISABLE_F9",None)
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"; SCENE_CUDA_DIR="/root/epic05_build_cache/scene/gsplat_scene_cuda"
for p in (C0_WT, SCENE_CUDA_DIR):
    if p not in sys.path: sys.path.insert(0,p)
import importlib.util
def load_ext(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name]=m; return m
load_ext("gsplat_cuda","/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so")
sys.modules["gsplat.csrc"]=sys.modules["gsplat_cuda"]
load_ext("experimental_gaussian_render_inference_scene_cuda","/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so")
sys.modules["gsplat.experimental.render.kernels.csrc"]=sys.modules["experimental_gaussian_render_inference_scene_cuda"]
from gsplat.experimental import rasterize_gaussian_higs_dynamic
sig=inspect.signature(rasterize_gaussian_higs_dynamic)
print("=== rasterize_gaussian_higs_dynamic params ===")
for n,p in sig.parameters.items():
    print("  ",n,"=",p.default)
print("=== source (first 60 lines) ===")
src=inspect.getsource(rasterize_gaussian_higs_dynamic)
print("\n".join(src.splitlines()[:60]))
