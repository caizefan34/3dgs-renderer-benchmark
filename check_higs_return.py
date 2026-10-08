#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
import inspect

sig = inspect.signature(rasterize_gaussian_higs_frozen)
print("Return annotation:", sig.return_annotation)
