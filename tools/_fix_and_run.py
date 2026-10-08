#!/usr/bin/env python3
"""
Run on mx via: python3 fix_and_run.py
This script patches _backend.py for C++14, sets env, and runs the timing.
"""
import os, sys, shutil, subprocess

# Ensure PATH
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':' + os.environ.get('PATH', '')
os.environ['PYTHONPATH'] = os.path.expanduser('~/.local/lib/python3.10/site-packages')
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ['CUDAHOSTCXX'] = '/usr/bin/g++-11'
os.environ['CXXFLAGS'] = '-std=c++14'

# Clean JIT cache
cache = os.path.expanduser('~/.cache/torch_extensions')
if os.path.exists(cache):
    shutil.rmtree(cache)
    print('Cleared JIT cache')

import gsplat
import gsplat.cuda._backend as be

# Patch _backend.py: force c++14
b_path = os.path.join(os.path.dirname(gsplat.__path__[0]), 'gsplat', 'cuda', '_backend.py')
with open(b_path) as f:
    bc = f.read()

bc = bc.replace('-std=c++17', '-std=c++14')
with open(b_path, 'w') as f:
    f.write(bc)
print('Patched _backend.py: c++17 -> c++14')

# Now run the timing script
script = os.path.expanduser('~/p5_baseline_runner.py')
print(f'Running: {script}')
subprocess.run([sys.executable, '-u', script], env=os.environ)
