import gsplat, inspect, json
# Check for internal _C, csrc modules
for name in sorted(dir(gsplat)):
    if not name.startswith('_') or name.startswith('__'):
        continue
    obj = getattr(gsplat, name, None)
    if obj is None:
        continue
    mt = type(obj).__name__
    members = [x for x in dir(obj) if not x.startswith('_')]
    if members:
        print(f"{name}: {mt}  members={members[:30]}")

print("---")
src = inspect.getsource(gsplat.rasterize_to_pixels)
if "last_ids" in src:
    print("last_ids FOUND in rasterize_to_pixels source")
    for line in src.split("\n"):
        if "last" in line.lower():
            print("  ", line.strip())
else:
    print("last_ids NOT in rasterize_to_pixels source")
    for line in src.split("\n")[-15:]:
        print("  ", line.strip())

# Also check what rasterize_to_pixels returns via its apply class
print("---checking _RasterizeToPixels---")
for n in dir(gsplat):
    if "Rasterize" in n and "Pixels" in n:
        obj = getattr(gsplat, n, None)
        if obj is not None:
            print(f"{n}: {type(obj).__name__}")
            # check for forward function
            if hasattr(obj, "forward"):
                fsrc = inspect.getsource(obj.forward)
                if "last_ids" in fsrc:
                    print("  forward contains last_ids!")
                else:
                    print("  forward does NOT contain last_ids")
