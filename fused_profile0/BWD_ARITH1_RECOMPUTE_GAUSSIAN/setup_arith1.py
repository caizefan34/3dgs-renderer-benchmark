import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

module_root = os.path.dirname(os.path.abspath(__file__))
oracle = os.environ.get('ARITH0_ORACLE', '0')

sources = [
    os.path.join(module_root, 'torch_bindings', 'bindings.cpp'),
    os.path.join(module_root, 'rasterization', 'src', 'forward.cu'),
    os.path.join(module_root, 'rasterization', 'src', 'backward.cu'),
    os.path.join(module_root, 'rasterization', 'src', 'bwd1_cache_symbol.cu'),
    os.path.join(module_root, 'rasterization', 'src', 'rasterization_api.cu'),
]

include_dirs = [
    os.path.join(module_root, 'utils'),
    os.path.join(module_root, 'rasterization', 'include'),
]

ext = CUDAExtension(
    name='_C_arith0',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': ['-std=c++17', '-O3'],
        'nvcc': ['-std=c++17', '-O3', '-DARITH0_ORACLE=' + oracle],
    },
)

setup(
    name='FasterGSFusedCudaBackend_arith0',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
