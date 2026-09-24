# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts/phase-c29")
from _c29_training import GaussianModel, W, H, DEV
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
import torch

scene = load_ply("data/official/mipnerf360/room/point_cloud.ply", device=DEV)
cams = resize_cameras(load_cameras_from_json("data/official/mipnerf360/room/cameras.json", device=DEV), W, H)
cam = cams[0]

# Create reference model and render
m = GaussianModel(scene, 3, DEV)
gt = m.render(cam).detach().unsqueeze(0).permute(0, 3, 1, 2).contiguous()
print(f"gt.requires_grad = {gt.requires_grad}, gt.grad_fn = {gt.grad_fn}")
del m

# Create training model
model = GaussianModel(scene, 3, DEV)
opt = model.get_optimizer()

# Test iteration 0
print(f"\nStep 0: model.means.grad before = {model.means.grad}")
rc = model.render(cam).unsqueeze(0).permute(0, 3, 1, 2)
loss = (rc - gt).abs().mean() + 0.2 * (1.0 - min(1.0, 1.0))
print(f"loss.grad_fn = {loss.grad_fn}")
loss.backward()
print(f"model.means.grad after backward = {model.means.grad.norm().item():.6f}")

# Zero grad explicitly
model.means.grad = None
model.quats.grad = None
model.scales.grad = None
model.opacities.grad = None
model.shs.grad = None

# Dense step (step 0, won't trigger densification since ds=100)
res = model.densify_and_prune(0, gt=0.0002, po=0.005)
print(f"Densify result: {res}, model.n={model.n}")

opt.step()
opt.zero_grad(set_to_none=True)

# Test iteration 1
print(f"\nStep 1: model.means.grad before = {model.means.grad}")
rc = model.render(cam).unsqueeze(0).permute(0, 3, 1, 2)
loss = (rc - gt).abs().mean() + 0.2 * (1.0 - min(1.0, 1.0))
try:
    loss.backward()
    print(f"Step 1 backward OK, grad = {model.means.grad.norm().item():.6f}" if model.means.grad is not None else "Step 1 backward OK, grad=None")
except RuntimeError as e:
    print(f"Step 1 backward FAILED: {e}")
    # Try with retain_graph
    loss.backward(retain_graph=True)
    print(f"  retain_graph=True worked")
