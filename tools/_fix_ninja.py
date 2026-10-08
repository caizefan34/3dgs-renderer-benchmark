"""Fix build.ninja for CUDA 13.3 + MSVC compatibility.
Must use C++20 (torch headers require it), with /Zc:preprocessor for MSVC."""
import os, re

p = os.path.expandvars(r'%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\gsplat_cuda\build.ninja')

with open(p) as f:
    c = f.read()

# 1. Ensure C++20 (torch headers use designated initializers, bit-field init)
c = c.replace('-std=c++17', '-std=c++20')

# 2. Add CCCL ignore and /Zc:preprocessor if not present
if 'CCCL_IGNORE' not in c:
    c = c.replace('--expt-relaxed-constexpr',
                  '--expt-relaxed-constexpr -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING')
    print('Added CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING')

if '/Zc:preprocessor' not in c:
    c = c.replace('-Xcompiler /MD', '-Xcompiler /MD -Xcompiler /Zc:preprocessor')
    print('Added -Xcompiler /Zc:preprocessor')

c = c.replace('CXX_FLAGS = /Zc:preprocessor', 'CXX_FLAGS =')
if 'CXX_FLAGS =' in c:
    c = c.replace('CXX_FLAGS =', 'CXX_FLAGS = /Zc:preprocessor')
    print('Added /Zc:preprocessor to CXX_FLAGS')

with open(p, 'w') as f:
    f.write(c)

# Verify
for line in open(p).readlines():
    if 'cuda_cflags' in line:
        print(f'CUDA: {"-DCCCL" in line}, {"/Zc:preprocessor" in line}, {"c++20" in line}')
    if 'CXX_FLAGS' in line and line.strip():
        print(f'CXX: {line.strip()[:120]}')

print('\nDone.')
