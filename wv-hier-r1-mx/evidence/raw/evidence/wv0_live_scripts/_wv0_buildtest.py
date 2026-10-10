import torch
from gsplat import rasterization
print("forcing build done; rendering...")
N=300
means=torch.randn(N,3,device="cuda")*0.3; means[:,2]+=3
q=torch.randn(N,4,device="cuda"); q=q/q.norm(dim=-1,keepdim=True)
s=torch.rand(N,3,device="cuda")*0.05+0.02
op=torch.rand(N,device="cuda"); c=torch.rand(N,3,device="cuda")
v=torch.eye(4,device="cuda").unsqueeze(0).contiguous()
K=torch.tensor([[300.,0,96],[0,300.,96],[0,0,1]],device="cuda").unsqueeze(0).contiguous()
means.requires_grad_(True)
img,alpha,meta=rasterization(means,q,s,op,c,v,K,width=192,height=192,tile_size=16,packed=False,sh_degree=None,render_mode="RGB")
print("RENDER OK", tuple(img.shape), "meta keys", list(meta.keys()))
img.square().mean().backward()
print("GRAD OK", tuple(means.grad.shape), "gnorm", float(means.grad.norm()))
# also test sh_degree path with packed=False
csh=torch.rand(N,16,3,device="cuda")
means2=means.detach().clone().requires_grad_(True)
img2,_,meta2=rasterization(means2,q,s,op,csh,v,K,width=192,height=192,tile_size=16,packed=False,sh_degree=3,render_mode="RGB")
img2.square().mean().backward()
print("SH RENDER+GRAD OK", tuple(img2.shape), tuple(means2.grad.shape))
