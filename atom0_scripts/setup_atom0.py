import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

module_root = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(module_root, 'torch_bindings', 'bindings_atom0.cpp'),
    os.path.join(module_root, 'rasterization', 'src', 'forward.cu'),
    os.path.join(module_root, 'rasterization', 'src', 'atom0_api.cu'),
]

include_dirs = [
    os.path.join(module_root, 'utils'),
    os.path.join(module_root, 'rasterization', 'include'),
]

ext = CUDAExtension(
    name='FasterGSFusedCudaBackend._C_atom0',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': ['-std=c++17', '-O3'],
        'nvcc': ['-std=c++17', '-O3', '-use_fast_math'],
    },
)

setup(
    name='FasterGSFusedCudaBackend_atom0',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
