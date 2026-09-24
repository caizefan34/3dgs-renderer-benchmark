#!/usr/bin/env python3
"""Test rasterization meta — visibility from radii, footprint from means2d."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))

import torch
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

device = "cuda:0"
repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=device)
sfm_data = load_initial_checkpoint("room", repo_root, device=device)

N = sfm_data['xyz'].shape[0]
model = GaussianModel(num_points=N, sh_degree=0, max_sh_degree=3, device=device)
model.init_from_sfm(
    xyz=sfm_data['xyz'].clone().float().to(device),
    opacity_logit=torch.logit(torch.full((N,), 0.1, device=device)),
    scales_log=sfm_data['scales'].clone().float().to(device),
    rotations_raw=sfm_data['rotations'].clone().float().to(device),
    shs=sfm_data['shs'].clone().float().to(device),
)
model.set_sh_degree(0)

data = model.forward()
cam = dataset.get_camera(0)

r, alpha, meta = rasterization(
    means=data["xyz"], quats=data["rotations"], scales=data["scales"],
    opacities=data["opacity"], colors=data["shs"],
    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
    width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=model.sh_degree,
    radius_clip=0.0, eps2d=0.1, render_mode="RGB",
)

radii = meta["radii"][0]  # [N, 2]
means2d = meta["means2d"][0]  # [N, 2]
depths = meta["depths"][0]  # [N]

print(f"radii shape: {radii.shape}, dtype: {radii.dtype}")
print(f"radii sample (first 10): {radii[:10]}")
print(f"radii > 0 count: {(radii > 0).sum().item()} / {N}")
print(f"radii[:, 0] > 0: {(radii[:, 0] > 0).sum().item()}")
print(f"radii[:, 1] > 0: {(radii[:, 1] > 0).sum().item()}")
print(f"radii[:, 0] max: {radii[:, 0].max().item()}")
print(f"radii[:, 1] max: {radii[:, 1].max().item()}")

# Visibility: radii > 0 (at least one dimension)
visible_mask = (radii > 0).any(dim=-1)
print(f"\nVisible mask: {visible_mask.sum().item()} / {N}")

# Screen-space radius: max of the two radii
screen_radius = radii.max(dim=-1).values.float()  # [N]
screen_radius[~visible_mask] = 0
print(f"Screen radius (visible): mean={screen_radius[visible_mask].mean().item():.2f}, "
      f"median={screen_radius[visible_mask].median().item():.2f}, "
      f"max={screen_radius.max().item():.2f}")

# Projected area (pi * r^2) — proxy
projected_area = 3.14159 * screen_radius ** 2
print(f"Projected area (visible): mean={projected_area[visible_mask].mean().item():.2f} px^2, "
      f"median={projected_area[visible_mask].median().item():.2f} px^2")

# means2d: projected 2D positions
print(f"\nmeans2d (visible, first 5): {means2d[visible_mask][:5]}")
print(f"means2d range: x=[{means2d[visible_mask][:, 0].min().item():.1f}, {means2d[visible_mask][:, 0].max().item():.1f}], "
      f"y=[{means2d[visible_mask][:, 1].min().item():.1f}, {means2d[visible_mask][:, 1].max().item():.1f}]")
print(f"Image size: {cam.image_width}x{cam.image_height}")

# depths
print(f"\nDepths (visible): mean={depths[visible_mask].mean().item():.2f}, "
      f"min={depths[visible_mask].min().item():.2f}, max={depths[visible_mask].max().item():.2f}")

# tiles_per_gauss
tiles_per_gauss = meta["tiles_per_gauss"][0]
print(f"\nTiles per Gauss: mean={tiles_per_gauss[visible_mask].float().mean().item():.2f}, "
      f"max={tiles_per_gauss.max().item()}")
