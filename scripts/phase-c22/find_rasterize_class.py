import os, gsplat, inspect

gsplat_dir = os.path.dirname(gsplat.__file__)
print("gsplat dir:", gsplat_dir)

# Find _RasterizeToPixels - it's likely in cuda/ops.py or similar
for root, dirs, files in os.walk(gsplat_dir):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path) as fh:
                try:
                    content = fh.read()
                    if "_RasterizeToPixels" in content:
                        rel = os.path.relpath(path, gsplat_dir)
                        print(f"Found in {rel}")
                        # Show lines around the class definition
                        for i, line in enumerate(content.split("\n")):
                            if "_RasterizeToPixels" in line or "class Rasterize" in line:
                                start = max(0, i-2)
                                for j in range(start, min(len(content.split("\n")), i+5)):
                                    print(f"  {j}: {content.split(chr(10))[j]}")
                                print()
                except:
                    pass
