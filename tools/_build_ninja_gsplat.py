#!/usr/bin/env python3
"""
Manually fix and run the ninja build for gsplat_cuda.
Adds -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR to nvcc commands,
then invokes ninja to complete the build.
"""
import sys, os, subprocess, re

JIT_DIR = os.path.expandvars(
    r'%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\gsplat_cuda')

ninja_path = os.path.join(JIT_DIR, 'build.ninja')

# Read and modify build.ninja
with open(ninja_path, 'r') as f:
    content = f.read()

# Add the define to nvcc commands (lines starting with "  nvcc" or containing "nvcc")
# The define must be added as a -D flag in the compilation rules
find_flag = '-ccbin'
replacement = '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING -ccbin'

# Add to all nvcc compile commands in the FLAGS lines
# The .cu.o rules use nvcc with a FLAGS variable
token = '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'
if token not in content:
    # Add to the main FLAGS variable definition
    content = content.replace(
        '--expt-relaxed-constexpr',
        '--expt-relaxed-constexpr ' + token
    )
    with open(ninja_path, 'w') as f:
        f.write(content)
    print('Added CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING to build.ninja')
else:
    print('Define already present in build.ninja')

# Run ninja
print('Running ninja...')
result = subprocess.run(
    ['ninja', '-v', '-j', '8'],
    cwd=JIT_DIR,
    capture_output=True, text=True, timeout=600
)

print(result.stdout[-2000:])
if result.stderr:
    # Only print last 2000 chars of stderr too
    print('STDERR:', result.stderr[-2000:])
print(f'ninja exit code: {result.returncode}')

if result.returncode == 0:
    # Verify .pyd exists
    pyd_files = [f for f in os.listdir(JIT_DIR) if f.endswith('.pyd')]
    print(f'SUCCESS! Built: {pyd_files}')
    print(f'File size: {os.path.getsize(os.path.join(JIT_DIR, pyd_files[0])) / 1024 / 1024:.1f} MB')
else:
    sys.exit(1)
