from torch.utils.cpp_extension import verify_ninja_availability, is_ninja_available
print("is_ninja_available", is_ninja_available())
verify_ninja_availability(); print("NINJA OK")
import gsplat
from gsplat import rasterization
print("gsplat import OK", gsplat.__version__)
import torch
N=200
means=torch.randn(N,3,device="cuda")*0.3; means[:,2]+=3
q=torch.randn(N,4,device="cuda"); q=q/q.norm(dim=-1,keepdim=True)
s=torch.rand(N,3,device="cuda")*0.05+0.02
op=torch.rand(N,device="cuda"); c=torch.rand(N,3,device="cuda")
v=torch.eye(4,device="cuda").unsqueeze(0).contiguous()
K=torch.tensor([[200.,0,64],[0,200.,64],[0,0,1]],device="cuda").unsqueeze(0).contiguous()
img,meta=rasterization(means,q,s,op,c,v,K,width=128,height=128,tile_size=16,packed=True,render_mode="RGB")
means.requires_grad_(True)
img2,_=rasterization(means,q,s,op,c,v,K,width=128,height=128,tile_size=16,packed=True)
img2.square().mean().backward()
print("RENDER+GRAD OK", tuple(img.shape), "gnorm", float(means.grad.norm()))
