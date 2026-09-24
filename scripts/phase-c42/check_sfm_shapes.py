import sys, torch
from pathlib import Path
sys.path.insert(0, '/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7')
sys.path.insert(0, '/home/liaoyuanjun/3dgs-renderer-benchmark/src')
from gaussian_model import GaussianModel
from dataset import load_initial_checkpoint

sfm = load_initial_checkpoint('room', Path('/home/liaoyuanjun/3dgs-renderer-benchmark'), 'cuda')
print("=== SfM data keys ===")
for k, v in sfm.items():
    print(f"  {k}: shape={v.shape} dtype={v.dtype}")

n = sfm['xyz'].shape[0]
model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device='cuda')
model.init_from_sfm(
    xyz=sfm['xyz'],
    opacity_logit=torch.logit(torch.full((n, 1), 0.1, device='cuda')),
    scales_log=sfm.get('scales'),
    rotations_raw=sfm.get('rotations'),
    shs=sfm.get('shs'))
model.set_sh_degree(3)
data = model.forward()
print("\n=== Model forward data ===")
for k, v in data.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

# Check what fully_fused_projection expects
from gsplat import fully_fused_projection
import inspect
sig = inspect.signature(fully_fused_projection)
print("\n=== fully_fused_projection signature ===")
print(f"  {sig}")
