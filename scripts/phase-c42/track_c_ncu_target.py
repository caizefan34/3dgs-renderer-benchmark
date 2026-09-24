#!/usr/bin/env python3
"""
Track C: Nsight Compute backward kernel profiling.

Runs a single forward+backward iteration under ncu to collect:
- Instruction breakdown
- exp/special function utilization
- Memory throughput
- Atomic throughput
- Occupancy

The script renders one camera from room scene and does backward.
ncu will profile the backward kernel specifically.
"""
import sys, torch, json
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
import torch.nn.functional as F

torch.manual_seed(42)
repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")

dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device="cuda")
sfm_data = load_initial_checkpoint("room", repo_root, device="cuda")
n = sfm_data["xyz"].shape[0]

model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device="cuda")
model.init_from_sfm(
    xyz=sfm_data["xyz"],
    opacity_logit=torch.logit(torch.full((n, 1), 0.1, device="cuda")),
    scales_log=sfm_data.get("scales"),
    rotations_raw=sfm_data.get("rotations"),
    shs=sfm_data.get("shs"))
model.set_sh_degree(3)

cam = dataset.get_camera(0)
gt = dataset.get_gt_image(0)
viewmats = cam.viewmatrix.unsqueeze(0)
Ks = cam.K.unsqueeze(0)

# Warmup
for _ in range(3):
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    data = model.forward()
    r, a, l = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=viewmats, Ks=Ks, width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    loss = F.l1_loss(r, gt.unsqueeze(0))
    loss.backward()
torch.cuda.synchronize()

# Profiled iteration
for p in model.parameters():
    if p.grad is not None: p.grad = None
data = model.forward()
r, a, l = rasterization(
    means=data["xyz"], quats=data["rotations"], scales=data["scales"],
    opacities=data["opacity"], colors=data["shs"],
    viewmats=viewmats, Ks=Ks, width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
loss = F.l1_loss(r, gt.unsqueeze(0))
loss.backward()
torch.cuda.synchronize()
print(f"Done. GS={model.xyz.shape[0]}, loss={loss.item():.4f}")
