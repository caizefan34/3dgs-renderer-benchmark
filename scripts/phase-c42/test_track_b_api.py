import sys, torch
from pathlib import Path
sys.path.insert(0, '/home/liaoyuanjun/3dgs-renderer-benchmark/src')
sys.path.insert(0, '/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7')
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss
from gsplat import rasterization
import torch.nn.functional as F

repo_root = Path('/home/liaoyuanjun/3dgs-renderer-benchmark')
dataset = GTDataset(scene='room', repo_root=repo_root, resolution='1080p', device='cuda')
sfm_data = load_initial_checkpoint('room', repo_root, device='cuda')
n = sfm_data['xyz'].shape[0]
print(f"SfM points: {n}")

model = GaussianModel(num_points=n, sh_degree=0, max_sh_degree=3, device='cuda')
model.init_from_sfm(
    xyz=sfm_data['xyz'],
    opacity_logit=torch.logit(torch.full((n, 1), 0.1, device='cuda')),
    scales_log=sfm_data.get('scales'),
    rotations_raw=sfm_data.get('rotations'),
    shs=sfm_data.get('shs'))
model.set_sh_degree(0)
print(f"Model initialized: {model.xyz.shape}")

cam = dataset.get_camera(0)
gt = dataset.get_gt_image(0)
print(f"Camera: {cam.image_width}x{cam.image_height}")
print(f"GT: {gt.shape}")

data = model.forward()
print(f"Data keys: {list(data.keys())}")
print(f"  xyz: {data['xyz'].shape}")
print(f"  shs: {data['shs'].shape}")
print(f"  opacity: {data['opacity'].shape}")

r, a, l = rasterization(
    means=data['xyz'], quats=data['rotations'], scales=data['scales'],
    opacities=data['opacity'], colors=data['shs'],
    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
    width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=0,
    radius_clip=0.0, eps2d=0.1, render_mode='RGB')
print(f"Render: {r.shape}, alpha: {a.shape}")

loss = F.l1_loss(r, gt.unsqueeze(0))
print(f"L1 loss: {loss.item():.4f}")

# Test d_ssim_loss
dsim = d_ssim_loss(r[0].clamp(0,1), gt)
print(f"D-SSIM: {dsim.item():.4f}")

# Test backward
loss2 = (1-0.2)*F.l1_loss(r, gt.unsqueeze(0)) + 0.2*dsim
loss2.backward()
print(f"Backward OK, xyz grad: {model.xyz.grad is not None}")
print("ALL TESTS PASSED")
