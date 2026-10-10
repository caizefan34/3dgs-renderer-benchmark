import torch, gsplat, inspect
print("exe", __import__("sys").executable)
print("gsplat", gsplat.__version__, gsplat.__file__)
from gsplat import isect_tiles, rasterize_to_pixels, fully_fused_projection
print("=== isect_tiles ===", inspect.signature(isect_tiles))
print("=== rasterize_to_pixels ===", inspect.signature(rasterize_to_pixels))
print("=== fully_fused_projection ===", inspect.signature(fully_fused_projection))
# tiny smoke render
torch.manual_seed(0)
N=500
means=torch.randn(N,3,device="cuda")*0.5
means[:,2]+=3.0
quats=torch.randn(N,4,device="cuda"); quats=quats/quats.norm(dim=-1,keepdim=True)
scales=torch.rand(N,3,device="cuda")*0.05+0.02
opac=torch.rand(N,device="cuda")
colors=torch.rand(N,3,device="cuda")
viewmats=torch.eye(4,device="cuda")[:None].expand(1,-1,-1).contiguous()
Ks=torch.tensor([[200.,0,64],[0,200.,64],[0,0,1]],device="cuda")[None]
img,meta=gsplat.rasterization(means,quats,scales,opac,colors,viewmats,Ks,width=128,height=128,
    tile_size=16,packed=True,render_mode="RGB",eps2d=0.3)
print("img",img.shape,"meta keys",list(meta.keys()))
print("means2d", meta["means2d"].shape if "means2d" in meta else "NA")
