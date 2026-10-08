import os, sys
os.environ["TORCH_EXTENSIONS_DIR"] = "/root/gsc0_ext"
import torch
import torch.utils._cpp_extension_versioner as V
def _hashable(v):
    try:
        hash(v); return True
    except TypeError:
        return False
_orig = V.update_hash
V.update_hash = lambda s, v: _orig(s, v) if _hashable(v) else _orig(s, repr(v))
ACC_TREE = "/root/epic05_frozen/gsplat-true-accutile-v153"
sys.path.insert(0, ACC_TREE)
import gsplat
from gsplat import rasterization
N, K = 16, 1
means  = torch.randn(N, 3, device="cuda")
quats  = torch.randn(N, 4, device="cuda"); quats = quats/quats.norm(dim=-1,keepdim=True)
scales = torch.rand(N, 3, device="cuda")*0.1
opac   = torch.rand(N, device="cuda")
colors = torch.randn(N, K, 3, device="cuda")
viewmats = torch.eye(4, device="cuda").unsqueeze(0)
Ks = torch.tensor([[[50.,0,32],[0,50,32],[0,0,1]]], device="cuda")
print(">>> rasterizing (unbatched, accutile=False) ...", flush=True)
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
print("BUILD_DIR:", os.listdir("/root/gsc0_ext/gsplat_cuda")[:6], flush=True)
print(">>> SUCCESS", flush=True)