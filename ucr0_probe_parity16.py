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
def dir_deriv(lt):
    grads=torch.autograd.grad(lt,params,retain_graph=True,allow_unused=True)
    return sum(float((gi*vi).sum()) for gi,vi in zip(grads,dirs) if gi is not None)
errs=[]
for s in range(16):
    g=torch.Generator(device="cuda"); g.manual_seed(1000+s)
    dirs=[]
    for p,lr in zip(params,lrs):
        r=torch.randint(0,2,tuple(p.shape),generator=g,device="cuda",dtype=torch.int8).float().mul_(2).sub_(1)
        dirs.append(r*lr)
    Df=dir_deriv(L_frozen); Dd=dir_deriv(pix_loss.mean())
    errs.append(abs(Dd-Df)/abs(Df))
errs=np.array(errs)
print("16-dir rel err: median=%.3e p10=%.3e max=%.3e mean=%.3e"%(np.median(errs),np.percentile(errs,10),errs.max(),errs.mean()))
print("pass median<=1e-4:", np.median(errs)<=1e-4, "max<=5e-4:", errs.max()<=5e-4)
