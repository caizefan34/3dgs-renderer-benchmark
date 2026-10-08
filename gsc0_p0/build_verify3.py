import os, sys
os.environ["TORCH_EXTENSIONS_DIR"] = "/root/gsc0_ext2"
import torch
import torch.utils._cpp_extension_versioner as V
def _h(v):
    try: hash(v); return True
    except TypeError: return False
_o = V.update_hash
V.update_hash = lambda s, v: _o(s, v) if _h(v) else _o(s, repr(v))
sys.path.insert(0, "/root/gsc0_p0/gsplat_pkg")
import gsplat
from gsplat import rasterization
print("gsplat file:", gsplat.__file__, flush=True)
N, K = 16, 1
torch.manual_seed(0)
means  = torch.randn(N, 3, device="cuda").requires_grad_(True)
quats  = torch.randn(N, 4, device="cuda").requires_grad_(True); quats = quats/quats.norm(dim=-1,keepdim=True)
scales = torch.rand(N, 3, device="cuda")*0.1
opac   = torch.rand(N, device="cuda").requires_grad_(True)
colors = torch.randn(N, K, 3, device="cuda").requires_grad_(True)
viewmats = torch.eye(4, device="cuda").unsqueeze(0)
Ks = torch.tensor([[[50.,0,32],[0,50,32],[0,0,1]]], device="cuda")
r, _, meta = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
    colors=colors, viewmats=viewmats, Ks=Ks, width=64, height=64, tile_size=16,
    packed=False, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    absgrad=True, accutile=False)
print("RASTERIZE OK img", tuple(r.shape), flush=True)
print("meta keys:", sorted(meta.keys()), flush=True)
print("has means2d/radii/tiles_per_gauss:", all(k in meta for k in ["means2d","radii","tiles_per_gauss"]), flush=True)
r.sum().backward()
print("BACKWARD OK; means.grad None?", means.grad is None, flush=True)
print("absgrad populated?", hasattr(means, "absgrad") and means.absgrad is not None, flush=True)
r2, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
    colors=colors, viewmats=viewmats, Ks=Ks, width=64, height=64, tile_size=16,
    packed=False, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    absgrad=True, accutile=True)
print("ACUTILE=True OK img", tuple(r2.shape), flush=True)
print(">>> SUCCESS", flush=True)