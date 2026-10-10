#!/usr/bin/env python3
"""Identify WHICH gsplat is actually providing each submodule, and whether
the runtime isect_tiles supports tile_mask."""
import inspect
from gsplat.cuda._wrapper import isect_tiles

print("=== runtime isect_tiles identity ===")
print("module file:", isect_tiles.__module__ if hasattr(isect_tiles, '__module__') else '?')
import gsplat.cuda._wrapper as w
print("_wrapper file:", w.__file__)
sig = inspect.signature(isect_tiles)
print("isect_tiles params:", list(sig.parameters.keys()))
print("HAS tile_mask:", "tile_mask" in sig.parameters)

# Now compare with the HiGS source copy
print("\n=== HiGS source _wrapper isect_tiles ===")
import importlib.util, os
src_wrapper = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/cuda/_wrapper.py"
print("source wrapper exists:", os.path.exists(src_wrapper))
with open(src_wrapper) as f:
    for i, line in enumerate(f, 1):
        if 'def isect_tiles' in line:
            print(f"  line {i}: {line.rstrip()}")
            # print signature region
            break

# Check the two _wrapper files' hashes
import hashlib
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
print("\n=== _wrapper.py identity ===")
print("runtime _wrapper sha256:", sha(w.__file__))
print("source  _wrapper sha256:", sha(src_wrapper))
print("MATCH:", sha(w.__file__) == sha(src_wrapper))
