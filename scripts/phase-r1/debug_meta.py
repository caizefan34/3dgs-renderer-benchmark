import sys, os, torch
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization
from dataset import GTDataset
import numpy as np

config = ReferenceV1Config(scene="room", iterations=30000)
dataset = GTDataset(config.scene, config.repo_root)
cam, gt = dataset.get_item(0)
ckpt = torch.load("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/ckpt_14k/checkpoints/iter_14000.pt", map_location="cpu", weights_only=False)
model = GaussianModel(max_sh_degree=config.sh_degree)
model.restore(ckpt, {
    "position_lr_init": config.position_lr_init, "position_lr_final": config.position_lr_final,
    "position_lr_delay_mult": config.position_lr_delay_mult, "position_lr_max_steps": config.position_lr_max_steps,
    "feature_lr": config.feature_lr, "opacity_lr": config.opacity_lr,
    "scaling_lr": config.scaling_lr, "rotation_lr": config.rotation_lr,
    "percent_dense": config.percent_dense,
})

r, _, meta = rasterization(
    means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
    opacities=model.get_opacity, colors=model.get_features,
    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
    width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=model.active_sh_degree,
    radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
)

print(f"image shape: {r.shape}")
print(f"radii shape: {meta['radii'].shape}")
print(f"radii ndim: {meta['radii'].ndim}")
print(f"tiles_per_gauss shape: {meta['tiles_per_gauss'].shape}")
print(f"meta keys: {list(meta.keys())}")
print(f"means2d shape: {meta['means2d'].shape}")
