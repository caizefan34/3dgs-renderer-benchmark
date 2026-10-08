"""Fix gsplat _backend.py for PyTorch 2.7.1 + GCC 11 + nvcc 11.5."""
import gsplat, os, shutil

pkg = os.path.dirname(gsplat.__path__[0])
bp = os.path.join(pkg, 'gsplat', 'cuda', '_backend.py')
with open(bp) as f:
    c = f.read()

mods = []

# 1. Fix _jit_compile API mismatch
old_sig = """            compiled = _jit_compile(
                name,
                sources,
                extra_cflags,
                extra_cuda_cflags,
                extra_ldflags,
                extra_include_paths,
                build_directory,
                verbose,
                with_cuda=None,"""

new_sig = """            compiled = _jit_compile(
                name,
                sources,
                extra_cflags,
                extra_cuda_cflags,
                extra_ldflags,
                extra_include_paths,
                build_directory,
                verbose,
                with_cuda=None,
                with_sycl=None,"""

if old_sig in c:
    c = c.replace(old_sig, new_sig)
    mods.append('_jit_compile API')
else:
    # fallback: try to add with_sycl after with_cuda
    if 'with_cuda=None' in c and 'with_sycl' not in c:
        c = c.replace('with_cuda=None,', 'with_cuda=None, with_sycl=None,')
        mods.append('_jit_compile API (fallback)')

# 2. Fix -std=c++17 -> -std=c++14 for all rules
c = c.replace('-std=c++17', '-std=c++14')
mods.append('c++17->c++14')

# 3. Remove -Wno-attributes (GCC-only flag that fails on some setups)
c = c.replace('-Wno-attributes', '')
mods.append('removed -Wno-attributes')

# 4. Add TORCH_CUDA_ARCH_LIST override to only build sm_80
c = c.replace(
    "extra_include_paths = [os.path.join(PATH, \"include/\"), glm_path]",
    "extra_include_paths = [os.path.join(PATH, \"include/\"), glm_path]\n        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')"
)
mods.append('arch_list=8.0')

with open(bp, 'w') as f:
    f.write(c)
print('Modifications:', mods)

# Clear JIT cache
cache = os.path.expanduser('~/.cache/torch_extensions')
if os.path.exists(cache):
    shutil.rmtree(cache)
    print('Cleared JIT cache')

print('Done. Ready to import gsplat.')
