#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

try:
    from gsplat.experimental.render.functional import gaussian_inference
    print("gaussian_inference exports:")
    print([x for x in dir(gaussian_inference) if not x.startswith('_')])
except Exception as e:
    print(f"Error: {e}")

# Check rasterization function
from gsplat import rasterization
print(f"\nrasterization signature:")
import inspect
print(inspect.signature(rasterization))
