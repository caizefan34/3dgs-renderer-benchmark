import sys, math
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
C.bootstrap_c0()
import torch, torch.nn.functional as F
import numpy as np
from fused_ssim import FusedSSIMMap
C1=0.01**2; C2=0.03**2; LAM=0.2
scene, it = "room", 10000
ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt); C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(5)
img,_,_ = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0,3,1,2).contiguous()
g = gt.unsqueeze(0).permute(0,3,1,2).contiguous()
H,W = b.shape[-2],b.shape[-1]
fused = C.make_fused_loss(); L_frozen = fused(img,gt)
ssim_map = FusedSSIMMap.apply(C1,C2,b,g,"same",True,2)
ssim_pix = ssim_map.mean(dim=1,keepdim=True)
l1_map = (b-g).abs().mean(dim=1,keepdim=True)
pix_loss = (1.0-LAM)*l1_map + LAM*(1.0-ssim_pix)
params = [getattr(model,n) for n in C.PARAM_NAMES]
def lr_of(p):
    for pg in model.optimizer.param_groups:
        if any(p is q for q in pg["params"]): return pg["lr"]
lrs=[lr_of(p) for p in params]
# total gradient relative error (optimizer units)
gf = torch.autograd.grad(L_frozen, params, retain_graph=True)
gd = torch.autograd.grad(pix_loss.mean(), params, retain_graph=True)
num=0.0; den=0.0
for a,bb,lr in zip(gf,gd,lrs):
    af=a.detach()*lr; bfd=bb.detach()*lr
    num += float((af-bfd).pow(2).sum()); den += float(af.pow(2).sum())
print("total grad rel err (opt units):", math.sqrt(num/den))
# cosine
dot=sum(float((a.detach()*bb.detach()).sum()) for a,bb in zip(gf,gd))
na=math.sqrt(sum(float(a.detach().pow(2).sum()) for a in gf))
nb=math.sqrt(sum(float(bb.detach().pow(2).sum()) for bb in gd))
print("global cosine:", dot/(na*nb))
# 16 directions: report |D_frozen|, |D_decomp|, rel err
rows=[]
for s in range(16):
    gg=torch.Generator(device="cuda"); gg.manual_seed(1000+s)
    dirs=[]
    for p,lr in zip(params,lrs):
        r=torch.randint(0,2,tuple(p.shape),generator=gg,device="cuda",dtype=torch.int8).float().mul_(2).sub_(1)
        dirs.append(r*lr)
    def dd(lt):
        gr=torch.autograd.grad(lt,params,retain_graph=True,allow_unused=True)
        return sum(float((gi*vi).sum()) for gi,vi in zip(gr,dirs) if gi is not None)
    Df=dd(L_frozen); Dd=dd(pix_loss.mean())
    rows.append((s,Df,Dd,abs(Dd-Df)/abs(Df) if abs(Df)>0 else float('nan')))
for s,Df,Df2,re in rows:
    print("dir %2d Dfrozen=%+.3e Ddecomp=%+.3e rel=%.3e"%(s,Df,Df2,re))
