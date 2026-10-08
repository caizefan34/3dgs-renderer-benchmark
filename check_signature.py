#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx")

from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
import inspect

sig = inspect.signature(rasterize_gaussian_higs_frozen)
print("Function signature:")
print(f"rasterize_gaussian_higs_frozen{sig}")
print("\nParameters:")
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'} = {param.default if param.default != inspect.Parameter.empty else 'required'}")
