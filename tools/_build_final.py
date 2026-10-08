"""Run ninja build for gsplat_cuda with fixes."""
import os, subprocess, re

JIT_DIR = os.path.expandvars(
    r'%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\gsplat_cuda')

# Check current CXX_FLAGS
ninja_path = os.path.join(JIT_DIR, 'build.ninja')

# First ensure build.ninja has the right flags
with open(ninja_path) as f:
    c = f.read()

# Apply all fixes idempotently
mods = []

# C++20
if '-std=c++17' in c:
    c = c.replace('-std=c++17', '-std=c++20')
    mods.append('c++17->c++20')

# CCCL_IGNORE
token = 'CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'
if token not in c:
    c = c.replace('--expt-relaxed-constexpr',
                  '--expt-relaxed-constexpr -D' + token)
    mods.append('CCCL_IGNORE')

# /Zc:preprocessor for CUDA
if '-Xcompiler /Zc:preprocessor' not in c:
    c = c.replace('-Xcompiler /MD', '-Xcompiler /MD -Xcompiler /Zc:preprocessor')
    mods.append('/Zc:preprocessor CUDA')

# /Zc:preprocessor for CXX
c = c.replace('CXX_FLAGS = /Zc:preprocessor', 'CXX_FLAGS =')
if 'CXX_FLAGS =' in c:
    c = c.replace('CXX_FLAGS =', 'CXX_FLAGS = /Zc:preprocessor')
    # Ensure not duplicated
    c = c.replace('/Zc:preprocessor /Zc:preprocessor', '/Zc:preprocessor')
    mods.append('/Zc:preprocessor CXX')

with open(ninja_path, 'w') as f:
    f.write(c)
print('Modifications:', mods)

# Run ninja
print('\nRunning ninja...')
result = subprocess.run(
    ['ninja', '-v', '-j', '8'],
    cwd=JIT_DIR,
    capture_output=True, text=True, timeout=600
)

output = result.stdout + result.stderr

# Find actual errors
errors = []
failed = []
for l in output.split('\n'):
    if 'error' in l.lower() and 'warning' not in l.lower() and 'errno' not in l.lower() and 'error_code' not in l.lower():
        if l.strip():
            errors.append(l.strip())
    if l.startswith('FAILED:'):
        failed.append(l)

print(f'\nExit code: {result.returncode}')
print(f'\nFailed targets ({len(failed)}):')
for f_ in failed:
    print(f'  {f_[:100]}')
print(f'\nActual errors ({len(errors)}):')
for e in errors[:20]:
    print(f'  {e[:200]}')

if result.returncode == 0:
    print('\n=== BUILD SUCCESS ===')
    pyds = [f for f in os.listdir(JIT_DIR) if f.endswith('.pyd')]
    print(f'Built: {pyds}')
else:
    print(f'\nLast 50 lines of output:')
    lines = output.split('\n')
    for l in lines[-50:]:
        if l.strip():
            print(f'  {l[:200]}')
