"""Examine build.ninja to find CXX compilation rules and flags."""
import os, re

p = os.path.expandvars(r'%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\gsplat_cuda\build.ninja')

with open(p) as f:
    c = f.read()

# Show all flag-like definitions
for line in open(p).readlines():
    if line.strip() and ('flag' in line.lower() or 'FLAGS' in line or 'CXX' in line or 'compile' in line.lower() or 'cuda_' in line.lower() or 'post_cflags' in line.lower()):
        print(line.rstrip()[:200])

print()
# Find the rule for .o targets (CXX compilation)
# Look for 'rule CXX' or 'command = $cxx' or 'cl'
for line in open(p).readlines():
    if 'rule ' in line.lower() or ('command =' in line and 'nvcc' not in line):
        print(line.rstrip()[:200])

print()
# Find lines with /Wno-attributes
for line in open(p).readlines():
    if 'Wno' in line:
        print(line.rstrip()[:200])
