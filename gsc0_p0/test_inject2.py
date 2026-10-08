import sys, os, types, importlib.util, traceback
ACC_TREE = "/root/epic05_frozen/gsplat-true-accutile-v153"
B1A_SO   = "/root/epic05_build_cache/b1a_ext/gsplat_cuda_final30k/gsplat_cuda_final30k.so"
sys.path.insert(0, ACC_TREE)
import gsplat  # true_accutile package (does NOT import _backend at top level)

# load prebuilt .so under its REAL module name so PyInit_gsplat_cuda is found
spec = importlib.util.spec_from_file_location("gsplat_cuda", B1A_SO)
_C = importlib.util.module_from_spec(spec)
sys.modules["gsplat_cuda"] = _C
spec.loader.exec_module(_C)
exported = sorted(a for a in dir(_C) if not a.startswith("_"))
print("prebuilt gsplat_cuda loaded; exported:", len(exported))
need = ["fully_fused_projection","rasterize_to_pixels","rasterize_to_pixels_from_world_3dgs",
        "spherical_harmonics","world_to_cam","quat_scale_to_covar_preci","isect_tiles","isect_offset_encode"]
print("has needed:", {n: hasattr(_C,n) for n in need})
acc = [e for e in exported if "accutile" in e.lower() or "acc" in e.lower()]
print("accutile-ish exports:", acc[:15])

# inject as gsplat.cuda._backend so wrapper's `from ._backend import _C` finds it
fake = types.ModuleType("gsplat.cuda._backend")
fake._C = _C
sys.modules["gsplat.cuda._backend"] = fake

import torch
from gsplat import rasterization
N, K = 8, 1
means  = torch.randn(1, N, 3, device="cuda")
quats  = torch.randn(1, N, 4, device="cuda"); quats = quats/quats.norm(dim=-1,keepdim=True)
scales = torch.rand(1, N, 3, device="cuda")*0.1
opac   = torch.rand(1, N, 1, device="cuda")
colors = torch.rand(1, N, K, 3, device="cuda")
viewmats = torch.eye(4, device="cuda")[:1].unsqueeze(0)
Ks = torch.tensor([[[50.,0,32],[0,50,32],[0,0,1]]], device="cuda")
try:
    r, meta, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors, viewmats=viewmats, Ks=Ks, width=64, height=64, tile_size=16,
        packed=False, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        absgrad=True, accutile=False)
    print("RASTERIZE OK img", tuple(r.shape), "meta", list(meta.keys()))
    # backward check
    r.sum().backward()
    print("BACKWARD OK; means.grad is None?", means.grad is None)
except Exception as e:
    print("RASTERIZE FAIL:", repr(e)[:300]); traceback.print_exc()