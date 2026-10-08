"""Fix gsplat _backend.py for PyTorch 2.7.1 + nvcc 11.5."""
import gsplat, os, shutil

bp = os.path.join(os.path.dirname(gsplat.__path__[0]), 'gsplat', 'cuda', '_backend.py')
with open(bp) as f:
    c = f.read()

# 1. Add PATH at top
c = c.replace('import os', 'import os\nos.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")')
# 2. Fix _jit_compile signature (add verbose, with_sycl defaults)
c = c.replace('def _jit_compile(', 'def _jit_compile(verbose=False, with_sycl=None, ')
# 3. Fix call site to include with_sycl
c = c.replace(
    '                verbose,\n                with_cuda=None,\n',
    '                verbose,\n                with_cuda=None,\n                with_sycl=None,\n'
)
# 4. C++14
c = c.replace('-std=c++17', '-std=c++14')
# 5. Remove -Wno-attributes
c = c.replace('-Wno-attributes', '')
# 6. Set arch list
c = c.replace(
    'extra_include_paths = [os.path.join(PATH, "include/"), glm_path]',
    'extra_include_paths = [os.path.join(PATH, "include/"), glm_path]\n        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")'
)

with open(bp, 'w') as f:
    f.write(c)

print('Backend patched successfully')
print('  c++17:', c.count('-std=c++17'), 'c++14:', c.count('-std=c++14'))

cache = os.path.expanduser('~/.cache/torch_extensions')
if os.path.exists(cache):
    shutil.rmtree(cache)
    print('JIT cache cleared')
