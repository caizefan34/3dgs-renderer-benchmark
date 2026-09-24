#!/usr/bin/env python3
"""Quick block-timing test to isolate CUDA event overhead. No events, no profiler."""
import sys, time, torch
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

torch.cuda.set_device(0)
torch.manual_seed(42)
dataset = GTDataset(scene="room", repo_root=".", resolution="1080p", device="cuda:0")
sfm_data = load_initial_checkpoint("room", ".", device="cuda:0")
n_cam = len(dataset)

model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device="cuda:0")
model.init_from_sfm(xyz=sfm_data["xyz"],
    opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device="cuda:0")),
    scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
sls = float(sfm_data["xyz"].norm(dim=-1).max().item())

opt = torch.optim.Adam([
    {"params": [model.xyz], "lr": 1.6e-4 * sls},
    {"params": [model.rotations], "lr": 1e-3},
    {"params": [model.scales], "lr": 5e-3},
    {"params": [model.opacity], "lr": 5e-2},
    {"params": [model.shs], "lr": 2.5e-3},
], eps=1e-15, betas=(0.9, 0.999))

# Warmup
for step in range(50):
    ci = step % n_cam
    cam = dataset.get_camera(ci)
    gt = dataset.get_gt_image(ci)
    data = model.forward()
    rendered, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"], viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=True, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        sparse_grad=False, absgrad=False)
    rendered = rendered[0].clamp(0, 1)
    loss = combined_loss(rendered, gt, lambda_dssim=0.2)["loss"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    model.accumulate_positional_gradient()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

# Clean block timing — NO CUDA events in measured loop
import gc
torch.cuda.synchronize()
block_start = time.perf_counter()
for step in range(50, 200):
    ci = step % n_cam
    cam = dataset.get_camera(ci)
    gt = dataset.get_gt_image(ci)
    data = model.forward()
    rendered, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"], viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=True, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        sparse_grad=False, absgrad=False)
    rendered = rendered[0].clamp(0, 1)
    loss = combined_loss(rendered, gt, lambda_dssim=0.2)["loss"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    model.accumulate_positional_gradient()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
torch.cuda.synchronize()
block_end = time.perf_counter()

iters = 150
total_ms = (block_end - block_start) * 1000
t_iter = total_ms / iters
print(f"Block: {iters} iters, total={total_ms:.1f}ms, T_iter={t_iter:.3f}ms")
