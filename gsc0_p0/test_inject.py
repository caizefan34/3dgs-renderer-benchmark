import sys, os, types, importlib.util, traceback
ACC_TREE = "/root/epic05_frozen/gsplat-true-accutile-v153"
B1A_SO   = "/root/epic05_build_cache/b1a_ext/gsplat_cuda_final30k/gsplat_cuda_final30k.so"
sys.path.insert(0, ACC_TREE)
print("torch", __import__("torch").__version__)

# 1) import the true_accutile gsplat Python package (does NOT import _backend yet)
import gsplat
print("gsplat package:", gsplat.__file__)

# what _wrapper needs from _C
import re
src = open(os.path.join(ACC_TREE, "gsplat", "cuda", "_wrapper.py")).read()
needed = sorted(set(re.findall(r"_C\.([A-Za-z0-9_]+)\(", src)))
print("_C symbols needed by _wrapper.py:", needed)

# 2) load the prebuilt accutile .so
spec = importlib.util.spec_from_file_location("gsplat_cuda_b1a_prebuilt", B1A_SO)
_C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_C)
exported = [a for a in dir(_C) if not a.startswith("__")]
print("prebuilt .so loaded; exported count:", len(exported))
missing = [s for s in needed if not hasattr(_C, s)]
print("MISSING symbols in prebuilt:", missing)

# 3) inject as gsplat.cuda._backend so _wrapper's `from ._backend import _C` finds it
fake = types.ModuleType("gsplat.cuda._backend")
fake._C = _C
sys.modules["gsplat.cuda._backend"] = fake

# 4) tiny rasterization smoke
import torch
from gsplat import rasterization
N, K = 8, 1
means   = torch.randn(1, N, 3, device="cuda")
quats   = torch.randn(1, N, 4, device="cuda"); quats = quats / quats.norm(dim=-1, keepdim=True)
scales  = torch.rand(1, N, 3, device="cuda") * 0.1
opac   = torch.rand(1, N, 1, device="cuda")
colors = torch.rand(1, N, K, 3, device="cuda")
viewmats = torch.eye(4, device="cuda")[:1].unsqueeze(0)
viewmats = torch.linalg.inv(viewmats)
Ks = torch.tensor([[[50.,0,32],[0,50,32],[0,0,1]]], device="cuda")
try:
    r, meta, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors, viewmats=viewmats, Ks=Ks, width=64, height=64, tile_size=16,
        packed=False, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        absgrad=True, accutile=False)
    print("RASTERIZE OK: img", tuple(r.shape), "meta keys", list(meta.keys())[:8])
except Exception as e:
    print("RASTERIZE FAIL:", repr(e)[:300])
    traceback.print_exc()