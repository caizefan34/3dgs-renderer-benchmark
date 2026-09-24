"""Check the gsplat rasterization pipeline sub-functions on this machine."""
import gsplat, inspect

print("=== Top-level functions ===")
for name in ["proj", "spherical_harmonics", "rasterize_to_pixels",
             "isect_tiles", "isect_offset_encode", "fully_fused_projection",
             "fully_fused_projection_with_ut", "sort_keys"]:
    fn = getattr(gsplat, name, None)
    if fn:
        try:
            sig = inspect.signature(fn)
            print(f"  {name}{sig}")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    else:
        print(f"  {name}: NOT AVAILABLE")

print()
print("=== rasterize_to_pixels internals ===")
src = inspect.getsource(gsplat.rasterization)
lines = src.split("\n")
for i, line in enumerate(lines):
    stripped = line.strip()
    if any(kw in stripped for kw in ["rasterize_to_pixels(", "isect_tiles(", "isect_offset_encode(",
                                      "fully_fused_projection(", "spherical_harmonics(",
                                      "proj(", "sort", "cub", "radix", "prefix"]):
        print(f"  L{i}: {stripped}")

print()
print("=== gsplat.cuda available ===")
cuda_dir = [x for x in dir(gsplat.cuda) if not x.startswith("_")]
print(f"  {cuda_dir}")

print()
print("=== gsplat.cuda._wrapper available ===")
wrapper_dir = [x for x in dir(gsplat.cuda._wrapper) if not x.startswith("_")]
print(f"  {wrapper_dir}")
