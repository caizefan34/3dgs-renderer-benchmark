import os, sys, types
os.environ["TORCH_EXTENSIONS_DIR"] = "/root/gsc0_ext2"
os.environ["MAX_JOBS"] = "8"
# versioner patch (torch 2.9.1 hash(list) crash)
import torch.utils._cpp_extension_versioner as V
def _h(v):
    try: hash(v); return True
    except TypeError: return False
_o = V.update_hash
V.update_hash = lambda s, v: _o(s, v) if _h(v) else _o(s, repr(v))
# use ISOLATED patched package copy
sys.path.insert(0, "/root/gsc0_p0/gsplat_pkg")
import gsplat
print("gsplat:", gsplat.__file__, "ver", gsplat.__version__, flush=True)
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
print(">>> building + rasterizing (accutile=False) ...", flush=True)
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
print("BUILD_DIR:", os.listdir("/root/gsc0_ext2/gsplat_cuda")[:5], flush=True)
print(">>> SUCCESS", flush=True)