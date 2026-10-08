import sys, math
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
C.bootstrap_c0()
import torch, torch.nn.functional as F
import fused_ssim, numpy as np
from fused_ssim import FusedSSIMMap

C1 = 0.01**2; C2 = 0.03**2; LAM = 0.2
scene, it = "room", 10000
ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(5)
img, _, _ = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0,3,1,2).contiguous()
g = gt.unsqueeze(0).permute(0,3,1,2).contiguous()
H, W = b.shape[-2], b.shape[-1]
fused = C.make_fused_loss()
L_frozen = fused(img, gt)

ssim_map = FusedSSIMMap.apply(C1, C2, b, g, "same", True, 2)
ssim_pix = ssim_map.mean(dim=1, keepdim=True)
l1_map = (b - g).abs().mean(dim=1, keepdim=True)
pix_loss = (1.0-LAM)*l1_map + LAM*(1.0 - ssim_pix)

# scalar parity: map reduction in float64
L_map = (pix_loss.double().sum()/(H*W))
print("scalar rel err (float64 map reduce):", abs(float(L_map)-float(L_frozen))/abs(float(L_frozen)))
# tile partition sum in float64
TS=16; nt_r=math.ceil(H/TS); nt_c=math.ceil(W/TS)
tot = 0.0
for tr in range(nt_r):
    for tc in range(nt_c):
        r0,c0=tr*TS,tc*TS
        tot += float(pix_loss[..., r0:r0+TS, c0:c0+TS].double().sum())
L_tile = tot/(H*W)
print("scalar rel err (float64 tile partition):", abs(L_tile-float(L_frozen))/abs(float(L_frozen)))

params = [getattr(model, n) for n in C.PARAM_NAMES]
def lr_of(p):
    for pg in model.optimizer.param_groups:
        if any(p is q for q in pg["params"]):
            return pg["lr"]
    return None
lrs = [lr_of(p) for p in params]
print("lrs", lrs)
g = torch.Generator(device="cuda"); g.manual_seed(123)
dirs = []
for p, lr in zip(params, lrs):
    r = torch.randint(0,2,tuple(p.shape), generator=g, device="cuda", dtype=torch.int8).float().mul_(2).sub_(1)
    dirs.append(r*lr)

def dir_deriv(loss_tensor):
    grads = torch.autograd.grad(loss_tensor, params, retain_graph=True, allow_unused=True)
    d=0.0
    for gi,vi in zip(grads,dirs):
        if gi is not None: d += float((gi*vi).sum())
    return d

D_frozen = dir_deriv(L_frozen)
D_map = dir_deriv(pix_loss.mean())
# tile-partition loss (float32 graph)
def tile_loss_fn():
    s = pix_loss[...,0:0,0:0].sum()*0.0
    for tr in range(nt_r):
        for tc in range(nt_c):
            r0,c0=tr*TS,tc*TS
            s = s + pix_loss[..., r0:r0+TS, c0:c0+TS].sum()/(H*W)
    return s
D_tile = dir_deriv(tile_loss_fn())
print("D_frozen", D_frozen)
print("D_map rel err", abs(D_map-D_frozen)/abs(D_frozen))
print("D_tile rel err", abs(D_tile-D_frozen)/abs(D_frozen))
