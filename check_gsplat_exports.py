#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

import gsplat
print("gsplat exports:")
print([x for x in dir(gsplat) if not x.startswith('_')])

# Check experimental
try:
    from gsplat.experimental import render
    print("\ngsplat.experimental.render exports:")
    print([x for x in dir(render) if not x.startswith('_')])
except Exception as e:
    print(f"\nError: {e}")
