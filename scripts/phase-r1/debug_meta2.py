import sys, os, torch, numpy as np
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")

from config import ReferenceV1Config
from gsplat import rasterization
from dataset import GTDataset

config = ReferenceV1Config(scene="room", iterations=30000)
dataset = GTDataset(config.scene, config.repo_root)
cam, gt = dataset.get_item(0)

# Load checkpoint and get model just for its tensors
ckpt = torch.load("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/ckpt_14k/checkpoints/iter_10000.pt", map_location="cuda", weights_only=False)
xyz = ckpt["xyz"].cuda()
shs = ckpt["shs"].cuda()
scaling = ckpt["scaling"].cuda()
rotation = ckpt["rotation"].cuda()
opacity = ckpt["opacity"].cuda()
print(f"Loaded N={xyz.shape[0]}")

# Run forward
sh_degree = ckpt["active_sh_degree"]
r, _, meta = rasterization(
    means=xyz, quats=rotation, scales=scaling,
    opacities=torch.sigmoid(opacity.squeeze(-1)),
    colors=shs,
    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
    width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=sh_degree,
    radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
)

print(f"image shape: {r.shape}")
print(f"radii shape: {meta['radii'].shape}")
print(f"radii ndim: {meta['radii'].ndim}")
print(f"tiles_per_gauss shape: {meta['tiles_per_gauss'].shape}")
print(f"means2d shape: {meta['means2d'].shape}")
print(f"meta keys: {list(meta.keys())}")
