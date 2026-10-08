#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
import inspect

sig = inspect.signature(rasterize_gaussian_higs_frozen)
print("Parameters that might affect memory:")
for name, param in sig.parameters.items():
    if 'tile' in name.lower() or 'cull' in name.lower() or 'sample' in name.lower() or 'compress' in name.lower():
        print(f"  {name}: default={param.default}")
