#!/usr/bin/env python3
"""Diagnostic: isolate backward cost by component. Fresh model.fwd each step."""
from __future__ import annotations
import sys, time, gc
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss, d_ssim_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint
import torch.nn.functional as F

device = "cuda"; torch.manual_seed(42)
dataset = GTDataset(scene="room", repo_root=ROOT, resolution="1080p", device=device)
sfm_data = load_initial_checkpoint("room", ROOT, device=device)
model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
model.init_from_sfm(xyz=sfm_data["xyz"],
    opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
    scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
print(f"Gs: {model.xyz.shape[0]:,}", flush=True)

cam0 = dataset.get_camera(0)
gt0 = dataset.get_gt_image(0)

def measure(warmup=8, n=15, label="", make_loss=None):
    """make_loss(rendered, gt) -> loss tensor. Fresh fwd each iteration."""
    vals = []
    for it in range(warmup + n):
        for p in model.parameters():
            p.grad = None
        data = model.forward()
        r, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=cam0.image_width, height=cam0.image_height,
            tile_size=16, packed=True, sh_degree=0,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        loss = make_loss(r[0].clamp(0,1), gt0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        if it >= warmup:
            vals.append(ms)
        gc.collect()
    m = float(np.mean(vals)) if vals else 0.0
    print(f"  {label}: mean={m:.3f}ms (n={len(vals)})")
    return m

print("\nMeasuring backward components...")
# Warmup once
data = model.forward()
r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"], opacities=data["opacity"], colors=data["shs"], viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0), width=cam0.image_width, height=cam0.image_height, tile_size=16, packed=True, sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
l = combined_loss(r[0].clamp(0,1), gt0)["loss"]
for p in model.parameters(): p.grad = None
l.backward()

raster_bwd = measure(make_loss=lambda r, g: r.float().mean(), label="Raster backward (dummy)")
l1_bwd = measure(make_loss=lambda r, g: F.l1_loss(r, g), label="L1 backward")
dsim_bwd = measure(make_loss=lambda r, g: d_ssim_loss(r, g), label="D-SSIM backward")
full_bwd = measure(make_loss=lambda r, g: combined_loss(r, g)["loss"], label="Full (L1+0.2*DSSIM) backward")

print(f"\nAnalysis:")
print(f"  Rasterization backward:  {raster_bwd:.3f}ms")
print(f"  + L1 grad computation:   {l1_bwd - raster_bwd:.3f}ms")
print(f"  + D-SSIM grad:           {dsim_bwd - raster_bwd:.3f}ms")
print(f"  + Autograd overhead:     {full_bwd - raster_bwd:.3f}ms")
print(f"  Full backward total:     {full_bwd:.3f}ms")
