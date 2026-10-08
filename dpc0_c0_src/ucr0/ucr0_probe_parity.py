import sys, math
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
C.bootstrap_c0()
import torch, torch.nn.functional as F
import fused_ssim, numpy as np
from fused_ssim import FusedSSIMMap

C1 = 0.01**2; C2 = 0.03**2; LAM = 0.2

# build a real model on one camera to get realistic preds + optimizer lrs
scene, it = "room", 10000
ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(5)
img, radii, means2d = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0,3,1,2).contiguous()
g = gt.unsqueeze(0).permute(0,3,1,2).contiguous()
H, W = b.shape[-2], b.shape[-1]
print("img", img.shape, "b", b.shape, "H*W", H*W)

# frozen fused loss (the reference)
fused = C.make_fused_loss()
L_frozen = fused(img, gt)
print("L_frozen", float(L_frozen))

# per-pixel loss map (unscaled): pix_loss.mean() == L_frozen
ssim_map = FusedSSIMMap.apply(C1, C2, b, g, "same", True, 2)   # (1,3,H,W)
ssim_pix = ssim_map.mean(dim=1, keepdim=True)                   # (1,1,H,W)
l1_map = (b - g).abs().mean(dim=1, keepdim=True)                # (1,1,H,W)
pix_loss = (1.0-LAM)*l1_map + LAM*(1.0 - ssim_pix)              # (1,1,H,W)
print("pix_loss.mean()", float(pix_loss.mean()), "rel err vs frozen",
      abs(float(pix_loss.mean())-float(L_frozen))/abs(float(L_frozen)))

# tile decomposition: 16x16 tiles, last row/col partial
TS = 16
nt_r = math.ceil(H/TS); nt_c = math.ceil(W/TS)
print("tiles", nt_r, "x", nt_c, "=", nt_r*nt_c)
# scalar: sum of tile losses (each = pix_loss over its 16x16 block, /(H*W))
L_decomp = 0.0
for tr in range(nt_r):
    for tc in range(nt_c):
        r0,c0 = tr*TS, tc*TS
        block = pix_loss[..., r0:r0+TS, c0:c0+TS]
        L_decomp = L_decomp + block.sum()/(H*W)
print("L_decomp", float(L_decomp), "rel err vs frozen",
      abs(float(L_decomp)-float(L_frozen))/abs(float(L_frozen)))

# gradient directional-derivative parity (16 Rademacher, optimizer-scaled)
params = [getattr(model, n) for n in C.PARAM_NAMES]
# lrs per group
lrs = []
for gname, pnames in C.GROUPS:
    p = getattr(model, pnames[0])
    lr = model.optimizer.param_groups[[pg for pg in model.optimizer.param_groups if p in pg['params'][::1] or True][0]] if False else None
# simpler: get lr from the param's group
def lr_of(p):
    for pg in model.optimizer.param_groups:
        if p in pg["params"]:
            return pg["lr"]
    return None
lrs = [lr_of(p) for p in params]
print("lrs", lrs)

g = torch.Generator(device="cuda"); g.manual_seed(123)
dirs = []
for p, lr in zip(params, lrs):
    r = torch.randint(0,2,(p.shape,), generator=g, device="cuda", dtype=torch.int8).float().mul_(2).sub_(1)
    dirs.append(r * lr)

def dir_deriv(loss_fn):
    grads = torch.autograd.grad(loss_fn, params, retain_graph=True, allow_unused=True)
    d = 0.0
    for gi, vi in zip(grads, dirs):
        if gi is not None:
            d += float((gi*vi).sum())
    return d

D_frozen = dir_deriv(L_frozen)
# decomposed: sum of per-tile grads dotted with dirs = grad of sum(tile_losses) dotted with dirs
def decomp_loss():
    s = pix_loss.new_zeros(())
    for tr in range(nt_r):
        for tc in range(nt_c):
            r0,c0 = tr*TS, tc*TS
            s = s + pix_loss[..., r0:r0+TS, c0:c0+TS].sum()/(H*W)
    return s
D_decomp = dir_deriv(decomp_loss())
print("D_frozen", D_frozen, "D_decomp", D_decomp, "rel err",
      abs(D_decomp-D_frozen)/abs(D_frozen))
