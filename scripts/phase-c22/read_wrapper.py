import os, gsplat
gsplat_dir = os.path.dirname(gsplat.__file__)
path = os.path.join(gsplat_dir, "cuda", "_wrapper.py")
with open(path) as f:
    lines = f.readlines()
# Print _RasterizeToPixels.forward lines 1250-1350
start = -1
for i, line in enumerate(lines):
    if "class _RasterizeToPixels" in line:
        start = i
    if start >= 0 and i > start and i < start + 120:
        print(f"{i}:{line}", end="")
