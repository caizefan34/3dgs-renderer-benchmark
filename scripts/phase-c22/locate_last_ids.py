import os, gsplat, inspect

# Find the autograd function by search - not imported at top level
src = inspect.getsource(gsplat.rasterize_to_pixels)
# Extract what is being called
for line in src.split("\n"):
    s = line.strip()
    if ".apply" in s or "Rasterize" in s or "autograd" in s:
        print(s)

print("===")
# Check the module for the function sources
gsplat_dir = os.path.dirname(gsplat.__file__)
print("gsplat dir:", gsplat_dir)

# Also check gsplat/rendering.py which is the high-level API
import os
rendering_path = os.path.join(gsplat_dir, "rendering.py")
if os.path.exists(rendering_path):
    with open(rendering_path) as f:
        c = f.read()
        if "last_ids" in c:
            print("last_ids IN rendering.py!")
            for l in c.split("\n"):
                if "last" in l.lower():
                    print(" ", l.strip())
        else:
            print("last_ids NOT in rendering.py")
            # show the rasterize_to_pixels call in rendering.py
            for l in c.split("\n"):
                if "rasterize_to_pixels" in l:
                    print(" ", l.strip())
