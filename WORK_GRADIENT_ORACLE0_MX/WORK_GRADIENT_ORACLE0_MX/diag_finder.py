#!/usr/bin/env python3
import os, sys, glob
print("=== meta_path ===")
for f in sys.meta_path:
    print("  ", f)
print("=== ENABLE_USER_SITE:", __import__("site").ENABLE_USER_SITE)
print("=== .pth files in higs env site-packages ===")
sp = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/lib/python3.10/site-packages"
for p in glob.glob(sp + "/*.pth"):
    print("  PTH:", os.path.basename(p))
    with open(p) as f:
        print("    ", f.read().strip()[:200])
print("=== sys.path (first 6) ===")
for p in sys.path[:6]:
    print("  ", p)

print("\n=== which gsplat resolves NOW ===")
import gsplat
print("gsplat.__file__:", gsplat.__file__)
print("gsplat.__path__:", list(gsplat.__path__))
