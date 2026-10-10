import gsplat, inspect
from gsplat import rasterization
print("=== rasterization signature ===")
print(inspect.signature(rasterization))
print("=== exported ===")
print([x for x in dir(gsplat) if not x.startswith("__")])
try:
    from gsplat import rasterize_to_indices_in_range
    print("=== rasterize_to_indices_in_range sig ===")
    print(inspect.signature(rasterize_to_indices_in_range))
except Exception as e:
    print("no indices_in_range:", e)
try:
    from gsplat import rasterize_pixels_to_pixels
    print("has rasterize_pixels_to_pixels")
except Exception as e:
    print("no pixels_to_pixels:", e)
import gsplat as g, os
print("gsplat file", g.__file__)
