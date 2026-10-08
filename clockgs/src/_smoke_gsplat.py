import torch, gsplat
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

torch.manual_seed(0)
dev="cuda"
N=200
means=torch.randn(N,3,device=dev)
scales=torch.log(torch.rand(N,3,device=dev)*0.1+0.01)
quats=torch.nn.functional.normalize(torch.randn(N,4,device=dev),dim=-1)
opacities=torch.rand(N,device=dev)
sh_degree=1
K=(sh_degree+1)**2
colors=torch.randn(N,K,3,device=dev)*0.1
C=2; H,W=64,64
Ks=torch.tensor([[[50.,0,W/2],[0,50.,H/2],[0,0,1]]],device=dev).expand(C,-1,-1).contiguous()
viewmats=torch.eye(4,device=dev).unsqueeze(0).expand(C,-1,-1).contiguous()
viewmats[:,2,3]=-4.0

colors_out, alphas, meta = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=viewmats, Ks=Ks, width=W, height=H, sh_degree=sh_degree,
    near_plane=0.01, far_plane=1000.0, packed=True, backgrounds=torch.zeros(3,device=dev),
    render_mode="RGB")
print("colors_out", colors_out.shape, "alphas", alphas.shape)
print("meta w/h/ncam", meta["width"], meta["height"], meta["n_cameras"])
print("has means2d/radii/conics/gaussian_ids:", "means2d" in meta, "radii" in meta, "conics" in meta, "gaussian_ids" in meta)

params={"means":means,"scales":scales,"quats":quats,"opacities":opacities,
        "sh0":colors[:,0:1].contiguous(),"shN":colors[:,1:].contiguous()}
opt={k: torch.optim.Adam([v], lr=1e-3) for k,v in params.items()}
strat=DefaultStrategy(refine_start_iter=5, refine_stop_iter=30, refine_every=2, verbose=False)
state=strat.initialize_state(scene_scale=1.0)
strat.check_sanity(params,opt)
gt=torch.rand(C,H,W,3,device=dev)
for step in range(12):
    for k in params: opt[k].zero_grad()
    colors=torch.cat([params["sh0"],params["shN"]],dim=1)
    colors_out, alphas, meta = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"],
        opacities=params["opacities"], colors=colors, viewmats=viewmats, Ks=Ks,
        width=W, height=H, sh_degree=sh_degree, near_plane=0.01, far_plane=1000.0,
        packed=True, backgrounds=torch.zeros(3,device=dev), render_mode="RGB")
    strat.step_pre_backward(params,opt,state,step,meta)
    loss=torch.nn.functional.l1_loss(colors_out,gt)
    loss.backward()
    strat.step_post_backward(params,opt,state,step,meta,packed=True)
    for k in params: opt[k].step()
print("OK strategy+train; N now", params["means"].shape[0])
