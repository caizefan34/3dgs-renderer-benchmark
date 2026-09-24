#!/usr/bin/env python3
"""Track C: nvprof-based kernel metrics for backward pass."""
import sys, torch, subprocess, json
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
import torch.nn.functional as F

torch.manual_seed(42)
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
dataset = GTDataset(scene="room", repo_root=repo, resolution="1080p", device="cuda")
sfm = load_initial_checkpoint("room", repo, device="cuda")
n = sfm["xyz"].shape[0]
model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device="cuda")
model.init_from_sfm(
    xyz=sfm["xyz"],
    opacity_logit=torch.logit(torch.full((n,1),0.1,device="cuda")),
    scales_log=sfm.get("scales"),
    rotations_raw=sfm.get("rotations"),
    shs=sfm.get("shs"))
model.set_sh_degree(3)
cam = dataset.get_camera(0)
gt = dataset.get_gt_image(0)
vm = cam.viewmatrix.unsqueeze(0)
K = cam.K.unsqueeze(0)

# warmup
for _ in range(5):
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    d = model.forward()
    r,a,l = rasterization(means=d["xyz"], quats=d["rotations"], scales=d["scales"],
        opacities=d["opacity"], colors=d["shs"], viewmats=vm, Ks=K,
        width=cam.image_width, height=cam.image_height, tile_size=16, packed=False,
        sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    loss = F.l1_loss(r, gt.unsqueeze(0))
    loss.backward()
torch.cuda.synchronize()

# profiled iteration
for p in model.parameters():
    if p.grad is not None: p.grad = None
d = model.forward()
r,a,l = rasterization(means=d["xyz"], quats=d["rotations"], scales=d["scales"],
    opacities=d["opacity"], colors=d["shs"], viewmats=vm, Ks=K,
    width=cam.image_width, height=cam.image_height, tile_size=16, packed=False,
    sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
loss = F.l1_loss(r, gt.unsqueeze(0))
loss.backward()
torch.cuda.synchronize()
print(f"Done. GS={n}, loss={loss.item():.4f}")
