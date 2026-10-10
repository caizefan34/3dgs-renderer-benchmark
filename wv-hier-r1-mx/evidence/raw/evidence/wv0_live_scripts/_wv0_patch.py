import torch, importlib.util, sys, os, types
so="/home/liaoyuanjun/warp_emit_a0/torch_extensions/gsplat_cuda/gsplat_cuda.so"
spec=importlib.util.spec_from_file_location("gsplat_cuda", so)
m=importlib.util.module_from_spec(spec)
sys.modules["gsplat_cuda"]=m
spec.loader.exec_module(m)
print("loaded prebuilt gsplat_cuda; has rasterize_to_pixels:", hasattr(m,"rasterize_to_pixels"), "isect_tiles:", hasattr(m,"isect_tiles"), "fully_fused_projection:", hasattr(m,"fully_fused_projection"))
# Pre-create a stub _backend module so gsplat's lazy 'from ._backend import _C' uses our prebuilt _C
stub=types.ModuleType("gsplat.cuda._backend")
stub._C=m
sys.modules["gsplat.cuda._backend"]=stub
from gsplat import rasterization
N=300
means=torch.randn(N,3,device="cuda")*0.3; means[:,2]+=3
q=torch.randn(N,4,device="cuda"); q=q/q.norm(dim=-1,keepdim=True)
s=torch.rand(N,3,device="cuda")*0.05+0.02
op=torch.rand(N,device="cuda"); c=torch.rand(N,3,device="cuda")
v=torch.eye(4,device="cuda").unsqueeze(0).contiguous()
K=torch.tensor([[300.,0,96],[0,300.,96],[0,0,1]],device="cuda").unsqueeze(0).contiguous()
means.requires_grad_(True)
img,meta=rasterization(means,q,s,op,c,v,K,width=192,height=192,tile_size=16,packed=False,sh_degree=None,render_mode="RGB")
print("RENDER OK", tuple(img.shape), "meta keys", list(meta.keys()))
img.square().mean().backward()
print("GRAD OK", tuple(means.grad.shape), "gnorm", float(means.grad.norm()))
