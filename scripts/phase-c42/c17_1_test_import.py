import gsplat
from gsplat import fully_fused_projection, isect_tiles
print('gsplat version:', gsplat.__version__)
print('Import test: PASS')
# Quick functional test
import torch
means2d = torch.randn(100, 2, device='cuda')
radii = torch.ones(100, 2, device='cuda', dtype=torch.int32) * 10
depths = torch.randn(100, device='cuda')
tpg, iids, fids = isect_tiles(means2d, radii, depths, 16, 120, 68, sort=True, packed=False)
print(f'isect_tiles test: PASS (n_isects={len(fids)})')
