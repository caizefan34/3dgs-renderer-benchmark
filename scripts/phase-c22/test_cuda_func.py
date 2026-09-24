import os, gsplat, inspect
gsplat_dir = os.path.dirname(gsplat.__file__)
from gsplat.cuda._wrapper import _make_lazy_cuda_func
print("_make_lazy_cuda_func imported OK")
# Verify the function works
func = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")
print(f"CUDA func: {func}")
sig = inspect.signature(func)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else '?'}")
print(f"Returns: {sig.return_annotation}")
