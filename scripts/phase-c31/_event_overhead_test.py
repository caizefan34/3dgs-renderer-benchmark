#!/usr/bin/env python3
"""Isolate CUDA event overhead: run block once without events, once with events."""
import sys, time, torch
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

def create_model_and_opt():
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
    return dataset, model, opt, n_cam

def run_one(dataset, model, opt, n_cam, step):
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

def warmup(dataset, model, opt, n_cam, steps=60):
    for step in range(steps):
        run_one(dataset, model, opt, n_cam, step)

import gc
torch.cuda.set_device(0)

# Test 1: Block timing WITHOUT CUDA events
dataset, model, opt, n_cam = create_model_and_opt()
warmup(dataset, model, opt, n_cam, 60)
torch.cuda.synchronize()
gc.collect()
b1 = time.perf_counter()
for step in range(60, 210):
    run_one(dataset, model, opt, n_cam, step)
torch.cuda.synchronize()
b1e = time.perf_counter()
t1 = (b1e - b1) * 1000 / 150
print(f"NO events: T_iter = {t1:.3f}ms")

# Test 2: Block timing WITH CUDA events (6 events per iter = 900 events)
dataset2, model2, opt2, n_cam2 = create_model_and_opt()
warmup(dataset2, model2, opt2, n_cam2, 60)
ev_fs = [torch.cuda.Event(enable_timing=True) for _ in range(150)]
ev_fe = [torch.cuda.Event(enable_timing=True) for _ in range(150)]
ev_bs = [torch.cuda.Event(enable_timing=True) for _ in range(150)]
ev_be = [torch.cuda.Event(enable_timing=True) for _ in range(150)]
ev_os = [torch.cuda.Event(enable_timing=True) for _ in range(150)]
ev_oe = [torch.cuda.Event(enable_timing=True) for _ in range(150)]
torch.cuda.synchronize()
gc.collect()
b2 = time.perf_counter()
for i in range(150):
    step = 60 + i
    ci = step % n_cam2
    cam = dataset2.get_camera(ci)
    gt = dataset2.get_gt_image(ci)
    data = model2.forward()
    ev_fs[i].record()
    rendered, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"], viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=True, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        sparse_grad=False, absgrad=False)
    ev_fe[i].record()
    rendered = rendered[0].clamp(0, 1)
    loss = combined_loss(rendered, gt, lambda_dssim=0.2)["loss"]
    opt2.zero_grad(set_to_none=True)
    ev_bs[i].record()
    loss.backward()
    ev_be[i].record()
    model2.accumulate_positional_gradient()
    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
    ev_os[i].record()
    opt2.step()
    ev_oe[i].record()
torch.cuda.synchronize()
b2e = time.perf_counter()
t2 = (b2e - b2) * 1000 / 150
print(f"WITH events (6 per iter): T_iter = {t2:.3f}ms")
print(f"Overhead per CUDA event: {(t2 - t1) / (150 * 6) * 1000:.1f}us")
