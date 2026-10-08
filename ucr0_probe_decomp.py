import sys
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
C.bootstrap_c0()
import torch, torch.nn.functional as F
import fused_ssim
from fused_ssim import FusedSSIMMap

torch.manual_seed(0)
pred = torch.rand(1,3,1080,1920, device="cuda")
tgt = torch.rand(1,3,1080,1920, device="cuda")

# frozen fused loss scalar
b = pred.contiguous(); g = tgt.contiguous()
L1 = F.l1_loss(b, g)
ssim_scalar = fused_ssim.fused_ssim(b, g, padding="same", train=True)
L_frozen = 0.8*L1 + 0.2*(1-ssim_scalar)
print("L_frozen", float(L_frozen), "L1", float(L1), "ssim", float(ssim_scalar))

# per-pixel ssim map
C1 = 0.01**2; C2 = 0.03**2
ssim_map, a, b2, c = fused_ssim.fusedssim(C1, C2, b, g, True)
print("ssim_map shape", ssim_map.shape, "dtype", ssim_map.dtype)
print("map.mean()", float(ssim_map.mean()), "vs fused_ssim", float(ssim_scalar))
print("map mean rel err", abs(float(ssim_map.mean())-float(ssim_scalar))/abs(float(ssim_scalar)))

# also via FusedSSIMMap.apply
m2 = FusedSSIMMap.apply(C1, C2, b, g, "same", True, 2)
print("FusedSSIMMap shape", m2.shape, "mean", float(m2.mean()))

# per-pixel l1 map (mean over channel) then spatial mean
l1_map = b.abs().sub(g).mean(dim=1, keepdim=True)  # (1,1,H,W)
print("l1_map spatial mean", float(l1_map.mean()), "vs L1", float(L1))
