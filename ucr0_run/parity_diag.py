import sys, math
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
C.bootstrap_c0()
import torch, torch.nn.functional as F
import numpy as np
from fused_ssim import FusedSSIMMap
C1=0.01**2; C2=0.03**2; LAM=0.2
def stable(s): import hashlib; return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],"little")%(2**63)
def decomp(b,g):
    sm=FusedSSIMMap.apply(C1,C2,b,g,"same",True,2).mean(dim=1,keepdim=True)
    lm=(b-g).abs().mean(dim=1,keepdim=True)
    return (1-LAM)*lm+LAM*(1-sm)
for cell in [("room",10000),("room",15000),("bicycle",10000),("bicycle",15000)]:
    scene,it=cell
    ckpt=C.load_checkpoint(scene,it); model=C.build_model(ckpt); C.scene_reset()
    ds=C.load_scene_dataset(scene)
    cam,gt=ds.get_item(39)
    img,_,_=C.render_view(model,cam,model.active_sh_degree)
    b=img.unsqueeze(0).permute(0,3,1,2).contiguous(); g=gt.unsqueeze(0).permute(0,3,1,2).contiguous()
    H,W=b.shape[-2],b.shape[-1]
    fused=C.make_fused_loss(); Lf=fused(img,gt)
    params=[getattr(model,n) for n in C.PARAM_NAMES]
    lrs=[]
    for p in params:
        for pg in model.optimizer.param_groups:
            if any(p is q for q in pg["params"]): lrs.append(pg["lr"]); break
    pix=decomp(b,g)
    scalar=abs(float(pix.double().sum()/(H*W))-float(Lf.detach()))/abs(float(Lf.detach()))
    def terr(a,b_):
        return math.sqrt(sum(float(((x-y)*lr).pow(2).sum()) for x,y,lr in zip(a,b_,lrs)))/math.sqrt(sum(float((x*lr).pow(2).sum()) for x,lr in zip(a,lrs)))
    gf1=torch.autograd.grad(Lf,params,retain_graph=True); gf2=torch.autograd.grad(Lf,params,retain_graph=True)
    gd=torch.autograd.grad(pix.mean(),params,retain_graph=True)
    noise=terr([x.detach() for x in gf1],[x.detach() for x in gf2])
    derr=terr([x.detach() for x in gd],[x.detach() for x in gf1])
    gnorm=math.sqrt(sum(float((x.detach()*lr).pow(2).sum()) for x,lr in zip(gf1,lrs)))
    errs=[]
    for s in range(16):
        gen=torch.Generator(device="cuda"); gen.manual_seed(stable("UCR0PARITY|%s@%d|%d"%(scene,it,s)))
        dirs=[]
        for p,lr in zip(params,lrs):
            r=torch.randint(0,2,tuple(p.shape),generator=gen,device="cuda",dtype=torch.int8).float().mul_(2).sub_(1); dirs.append(r*lr)
        def dd(lt):
            gr=torch.autograd.grad(lt,params,retain_graph=True,allow_unused=True)
            return sum(float((gi*vi).sum()) for gi,vi in zip(gr,dirs) if gi is not None)
        errs.append(abs(dd(pix.mean())-dd(Lf))/gnorm)
    errs=np.array(errs)
    print("%s@%d scalar=%.2e total_grad_err=%.3e noise=%.3e ratio=%.2f dir_med=%.3e dir_max=%.3e"%(
        scene,it,scalar,derr,noise,derr/noise,np.median(errs),errs.max()))
