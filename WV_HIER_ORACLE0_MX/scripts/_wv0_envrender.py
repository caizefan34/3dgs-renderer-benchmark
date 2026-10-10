import torch, gsplat as g, time
print("torch",torch.__version__,"cuda",torch.version.cuda,"gsplat",g.__version__,g.__file__)
N=200
means=torch.randn(N,3,device="cuda")*0.3; means[:,2]+=3
quats=torch.randn(N,4,device="cuda"); quats=quats/quats.norm(dim=-1,keepdim=True)
scales=torch.rand(N,3,device="cuda")*0.05+0.02
opac=torch.rand(N,device="cuda"); colors=torch.rand(N,3,device="cuda")
v=torch.eye(4,device="cuda").unsqueeze(0).contiguous()
K=torch.tensor([[200.,0,64],[0,200.,64],[0,0,1]],device="cuda").unsqueeze(0).contiguous()
img,meta=g.rasterization(means,quats,scales,opac,colors,v,K,width=128,height=128,tile_size=16,packed=True,render_mode="RGB")
print("RENDER OK",tuple(img.shape),"meta",list(meta.keys()))
means.requires_grad_(True)
img2,_=g.rasterization(means,quats,scales,opac,colors,v,K,width=128,height=128,tile_size=16,packed=True,render_mode="RGB")
img2.square().mean().backward()
print("grad OK", tuple(means.grad.shape), "norm", float(means.grad.norm()))
