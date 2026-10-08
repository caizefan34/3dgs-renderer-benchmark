"""Fix CXX_FLAGS in build.ninja - remove GCC flags MSVC doesn't understand."""
import os

JIT_DIR = os.path.expandvars(
    r'%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\gsplat_cuda')
ninja_path = os.path.join(JIT_DIR, 'build.ninja')

with open(ninja_path) as f:
    c = f.read()

# Show CXX-related lines
print("=== CXX rules ===")
for line in open(ninja_path).readlines():
    if 'CXX' in line and line.strip():
        print(line.rstrip()[:200])

# Find the CXX compilation rule
# The issue: CXX_FLAGS includes /Wno-attributes which MSVC rejects
# MSVC equivalent would be /wd5219 (warning number), but simplest is to remove it.

# Find CXX_FLAGS definition
import re
cxx_match = re.search(r'CXX_FLAGS = (.*)', c)
if cxx_match:
    cxx_flags = cxx_match.group(1)
    print(f'\nCurrent CXX_FLAGS: {cxx_flags}')
    
    # Remove /Wno-attributes (GCC flag, not MSVC)
    new_flags = cxx_flags.replace('/Wno-attributes', '')
    # Also remove double spaces
    new_flags = ' '.join(new_flags.split())
    
    if new_flags != cxx_flags:
        c = c.replace(cxx_flags, new_flags, 1)
        print(f'Fixed: removed /Wno-attributes')
        print(f'New CXX_FLAGS: {new_flags}')
    
    with open(ninja_path, 'w') as f:
        f.write(c)
    print('Saved build.ninja')
else:
    print('CXX_FLAGS not found!')
