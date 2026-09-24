import torch
from gsplat.cuda._backend import _C
print("_C loaded:", _C)
print("rasterize_to_pixels_3dgs_bwd:", hasattr(_C, "rasterize_to_pixels_3dgs_bwd"))
