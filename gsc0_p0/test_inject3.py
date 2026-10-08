import os, sys, types, importlib.util, traceback
ACC_TREE = "/root/epic05_frozen/gsplat-true-accutile-v153"
B1A_SO   = "/root/epic05_build_cache/b1a_ext/gsplat_cuda_final30k/gsplat_cuda_final30k.so"
sys.path.insert(0, ACC_TREE)
import gsplat
print("gsplat pkg:", gsplat.__file__, flush=True)

# load B1A accutile .so under its REAL module name
spec = importlib.util.spec_from_file_location("gsplat_cuda_final30k", B1A_SO)
_C = importlib.util.module_from_spec(spec)
sys.modules["gsplat_cuda_final30k"] = _C
spec.loader.exec_module(_C)
exported = sorted(a for a in dir(_C) if not a.startswith("_"))
print("B1A _C loaded; exported:", len(exported), flush=True)

# check symbols _wrapper.py needs (lazy getattr)
import re
src = open(os.path.join(ACC_TREE, "gsplat", "cuda", "_wrapper.py")).read()
# _wrapper uses a lazy loader; collect referenced attr names via the load table
need = ["fully_fused_projection","rasterize_to_pixels","rasterize_to_pixels_from_world_3dgs",
        "spherical_harmonics","world_to_cam","quat_scale_to_covar_preci","isect_tiles",
        "isect_offset_encode","rasterize_to_pixels_from_world_3dgs"]
print("has needed:", {n: hasattr(_C, n) for n in need}, flush=True)

# inject as gsplat.cuda._backend so `from ._backend import _C` resolves to our _C
fake = types.ModuleType("gsplat.cuda._backend")
fake._C = _C
sys.modules["gsplat.cuda._backend"] = fake

import torch
from gsplat import rasterization
N, K = 16, 1
torch.manual_seed(0)
means  = torch.randn(N, 3, device="cuda")
quats  = torch.randn(N, 4, device="cuda"); quats = quats/quats.norm(dim=-1,keepdim=True)
scales = torch.rand(N, 3, device="cuda")*0.1
opac   = torch.rand(N, device="cuda")
colors = torch.randn(N, K, 3, device="cuda")
viewmats = torch.eye(4, device="cuda").unsqueeze(0)
Ks = torch.tensor([[[50.,0,32],[0,50,32],[0,0,1]]], device="cuda")
print(">>> rasterizing accutile=False ...", flush=True)
r, meta, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
    colors=colors, viewmats=viewmats, Ks=Ks, width=64, height=64, tile_size=16,
    packed=False, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    absgrad=True, accutile=False)
print("RASTERIZE OK img", tuple(r.shape), "meta", list(meta.keys()), flush=True)
r.sum().backward()
print("BACKWARD OK; means.grad None?", means.grad is None, flush=True)
r2, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
    colors=colors, viewmats=viewmats, Ks=Ks, width=64, height=64, tile_size=16,
    packed=False, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    absgrad=True, accutile=True)
print("ACUTILE=True OK img", tuple(r2.shape), flush=True)
print(">>> SUCCESS", flush=True)