import inspect
from gsplat.rendering import rasterization
sig = inspect.signature(rasterization)
params = list(sig.parameters.keys())
print("rasterization params:", params)
print("has importance_mask:", "importance_mask" in params)
