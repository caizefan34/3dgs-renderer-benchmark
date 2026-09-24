#!/usr/bin/env python3
"""Add --extended-lambda to gsplat v1.5.3 build flags for AccuTile.
The AccuTile predicate uses a __device__ lambda which requires this flag."""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153"

# Fix setup.py
setup_path = f"{root}/setup.py"
with open(setup_path) as f:
    content = f.read()
old = 'nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]'
new = 'nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17", "--extended-lambda"]'
assert old in content, f"Cannot find expected text in setup.py"
content = content.replace(old, new, 1)
with open(setup_path, "w") as f:
    f.write(content)
print(f"  MODIFIED setup.py: added --extended-lambda")

# Fix _backend.py (JIT fallback)
backend_path = f"{root}/gsplat/cuda/_backend.py"
with open(backend_path) as f:
    content = f.read()
old = 'extra_cuda_cflags += ["-use_fast_math"]'
new = 'extra_cuda_cflags += ["-use_fast_math", "--extended-lambda"]'
if old in content:
    content = content.replace(old, new, 1)
    with open(backend_path, "w") as f:
        f.write(content)
    print(f"  MODIFIED _backend.py: added --extended-lambda")
else:
    print(f"  SKIP _backend.py (already modified or not found)")
