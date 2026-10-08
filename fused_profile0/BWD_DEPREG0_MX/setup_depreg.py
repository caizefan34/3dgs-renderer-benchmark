import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
module_root = os.path.dirname(os.path.abspath(__file__))
extra = os.environ.get("NVCC_EXTRA", "-Xptxas -v")
sources = [
    os.path.join(module_root, "torch_bindings", "bindings.cpp"),
    os.path.join(module_root, "rasterization", "src", "forward.cu"),
    os.path.join(module_root, "rasterization", "src", "backward.cu"),
    os.path.join(module_root, "rasterization", "src", "rasterization_api.cu"),
]
include_dirs = [os.path.join(module_root,"utils"), os.path.join(module_root,"rasterization","include")]
nvcc_args = ["-std=c++17","-O3"] + extra.split()
ext = CUDAExtension(name="_C_depreg0", sources=sources, include_dirs=include_dirs,
    extra_compile_args={"cxx":["-std=c++17","-O3"],"nvcc":nvcc_args})
setup(name="BWD_DEPREG0", ext_modules=[ext], cmdclass={"build_ext":BuildExtension})
