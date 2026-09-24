import importlib.util, torch
CORE="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
spec=importlib.util.spec_from_file_location("gsplat_cuda", CORE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
attrs=[a for a in dir(mod) if not a.startswith('__')]
print("MODULE ATTRS:", attrs)
try:
    opns=[a for a in dir(torch.ops.gsplat) if 'isect' in a or 'offset' in a or 'tile' in a]
    print("TORCH.OPS.GSPLAT isect/offset/tile:", opns)
except Exception as e:
    print("ops err", e)