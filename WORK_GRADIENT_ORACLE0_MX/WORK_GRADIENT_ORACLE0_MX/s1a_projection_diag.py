#!/usr/bin/env python3
"""Diagnostic: project checkpoint means into camera 0 and check pixel coverage.
Determines if means are in a renderable space or need normalization."""
import os, sys, json
import numpy as np, torch
dev="cuda:6"
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
SCENE = "/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means = g["gaussians._means"].to(dev).float().cpu().numpy()
print(f"N={len(means)}")
print(f"means x: [{means[:,0].min():.1f},{means[:,0].max():.1f}] mean {means[:,0].mean():.1f}")
print(f"means y: [{means[:,1].min():.1f},{means[:,1].max():.1f}] mean {means[:,1].mean():.1f}")
print(f"means z: [{means[:,2].min():.1f},{means[:,2].max():.1f}] mean {means[:,2].mean():.1f}")
print(f"means std: {means.std(0)}")

cams = json.load(open(f"{SCENE}/eval_cameras.json"))
c = cams[0]
R = np.asarray(c["rotation"],dtype=np.float64); p = np.asarray(c["position"],dtype=np.float64)
Rw2c = R.T
# transform means to camera space (as-is)
mc = (Rw2c @ (means - p).T).T  # [N,3] camera coords
print(f"\ncam-space (as-is): x[{mc[:,0].min():.1f},{mc[:,0].max():.1f}] y[{mc[:,1].min():.1f},{mc[:,1].max():.1f}] z[{mc[:,2].min():.1f},{mc[:,2].max():.1f}]")
print(f"  frac z>0 (in front, OpenCV +z): {(mc[:,2]>0).mean():.3f}")
print(f"  frac z<0 (behind): {(mc[:,2]<0).mean():.3f}")
# flip z (OpenGL)
mc_z = mc.copy(); mc_z[:,2]*=-1; mc_z[:,1]*=-1
print(f"cam-space (flip y,z): z[{mc_z[:,2].min():.1f},{mc_z[:,2].max():.1f}] frac z>0: {(mc_z[:,2]>0).mean():.3f}")

# Project to pixels (as-is, OpenCV +z)
fx,fy,cx,cy = float(c["fx"])*0.25, float(c["fy"])*0.25, (int(c["width"]*0.25)-1)/2, (int(c["height"]*0.25)-1)/2
W,H = int(c["width"]*0.25), int(c["height"]*0.25)
z = mc[:,2]
valid = z>0.1
u = fx*mc[:,0]/np.clip(z,1e-6,None)+cx
v = fy*mc[:,1]/np.clip(z,1e-6,None)+cy
inb = valid & (u>=0)&(u<W)&(v>=0)&(v<H)
print(f"\nprojection (as-is): {inb.sum()} gaussians in front & in-bounds / {len(means)}")
print(f"  u range [{u[valid].min():.0f},{u[valid].max():.0f}] v range [{v[valid].min():.0f},{v[valid].max():.0f}]")

# What if means need to be centered? check centroid
print(f"\nmeans centroid: {means.mean(0)}")
print(f"camera position: {p}")
# distance from camera to centroid
print(f"dist cam->centroid: {np.linalg.norm(means.mean(0)-p):.1f}")
