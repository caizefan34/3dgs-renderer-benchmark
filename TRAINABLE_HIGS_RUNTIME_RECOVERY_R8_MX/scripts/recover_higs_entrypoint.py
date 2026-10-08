#!/usr/bin/env python3
"""R8 P0: Recover R3 HiGS entrypoint"""
import sys
import os

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

print("=== R8 P0: HiGS Runtime Recovery ===\n")

# Try to import the exact function R3 used
print("1. Attempting to import rasterize_gaussian_higs_frozen...")
try:
    from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
    print("   ✓ SUCCESS - Function found")
    print(f"   Function: {rasterize_gaussian_higs_frozen}")
    print(f"   Module: {rasterize_gaussian_higs_frozen.__module__}")
except ImportError as e:
    print(f"   ✗ FAILED - {e}")
    print("\n2. Checking gsplat.experimental.render.functional...")
    try:
        import gsplat.experimental.render.functional as higs_func
        print(f"   Module loaded: {higs_func}")
        print(f"   Available functions: {[x for x in dir(higs_func) if not x.startswith('_')]}")
    except Exception as e2:
        print(f"   ✗ Module import failed: {e2}")
