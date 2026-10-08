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
def gnorm_opt(grads):
    return math.sqrt(sum(float((gr.detach()*lr).pow(2).sum()) for gr,lr in zip(grads,lrs)))
# noise floor: grad(L_frozen) three times
nf=[]
for _ in range(3):
    g1=torch.autograd.grad(L_frozen,params,retain_graph=True)
    nf.append(gnorm_opt(g1))
print("||g_frozen|| opt (3 runs):", nf)
# pairwise run-to-run rel
g1=torch.autograd.grad(L_frozen,params,retain_graph=True)
g2=torch.autograd.grad(L_frozen,params,retain_graph=True)
gd=torch.autograd.grad(pix_loss.mean(),params,retain_graph=True)
def rel(a,bb):
    num=math.sqrt(sum(float(((x-y)*lr).pow(2).sum()) for x,y,lr in zip(a,bb,lrs)))
    den=math.sqrt(sum(float((x*lr).pow(2).sum()) for x,lr in zip(a,lrs)))
    return num/den
print("run-to-run rel (frozen vs frozen):", rel(g1,g2))
print("decomp vs frozen rel:", rel(gd,g1))
# stable normalized directional error |dD|/||g_opt| for 16 dirs
errs=[]
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
    gnorm=gnorm_opt(g1)
    errs.append(abs(Dd-Df)/gnorm)
errs=np.array(errs)
print("|dD|/||g_opt||: median=%.3e max=%.3e"%(np.median(errs),errs.max()))
