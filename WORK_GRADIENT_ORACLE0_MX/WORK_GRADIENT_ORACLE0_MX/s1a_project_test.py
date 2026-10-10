#!/usr/bin/env python3
"""Manual projection test: do the gaussians land on-screen for camera 0?
Checks if the camera pose is sane independent of the rasterizer."""
import os, sys, json
import torch, numpy as np
dev = torch.device("cuda:6")
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means = g["gaussians._means"].to(dev).float()  # [N,3] world
N = means.shape[0]
quats = g["gaussians._rotations"].to(dev).float()
scales = g["gaussians._scales"].to(dev).float().exp()

cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W, H = 396, 264

for cam_idx in [0, 2]:
    c = cams[cam_idx]; scale = W/float(c["width"])
    R = np.asarray(c["rotation"], np.float64)  # c2w
    p = np.asarray(c["position"], np.float64)  # camera center
    Rw2c = R.T
    vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
    vm = torch.tensor(vm, dtype=torch.float32, device=dev)
    K = torch.tensor([[c["fx"]*scale,0,(W-1)/2.0],[0,c["fy"]*scale,(H-1)/2.0],[0,0,1.0]], dtype=torch.float32, device=dev)
    # transform means to camera space
    Rinv = vm[:3,:3].T  # back to world->? use vm directly
    cam_pts = (vm[:3,:3] @ means.T).T + vm[:3,3]  # [N,3] camera coords
    z = cam_pts[:,2]
    # project
    uv = (K @ cam_pts.T).T  # [N,3]
    u = uv[:,0]/uv[:,2].clamp_min(1e-6)
    v = uv[:,1]/uv[:,2].clamp_min(1e-6)
    in_front = (z > 0.2) & (z < 10000)
    on_screen = in_front & (u>=0) & (u<W) & (v>=0) & (v<H)
    print(f"\n=== camera {cam_idx} {c['img_name']} ===")
    print(f"means world range x[{means[:,0].min():.0f},{means[:,0].max():.0f}] y[{means[:,1].min():.0f},{means[:,1].max():.0f}] z[{means[:,2].min():.0f},{means[:,2].max():.0f}]")
    print(f"cam center world: {p}")
    print(f"cam-space z range: [{z.min():.2f},{z.max():.2f}]  median {z.median():.2f}")
    print(f"in-frustum(0.2<z<10000): {in_front.sum().item()}/{N} ({in_front.float().mean()*100:.1f}%)")
    print(f"on-screen (u,v in [0,W]x[0,H]): {on_screen.sum().item()} ({on_screen.float().mean()*100:.1f}%)")
    if on_screen.sum() > 0:
        print(f"on-screen u range [{u[on_screen].min():.0f},{u[on_screen].max():.0f}] v range [{v[on_screen].min():.0f},{v[on_screen].max():.0f}]")
        print(f"on-screen z median {z[on_screen].median():.2f}")
    # camera-space extent of the on-screen points
    print(f"all cam-space pts: x[{cam_pts[:,0].min():.1f},{cam_pts[:,0].max():.1f}] y[{cam_pts[:,1].min():.1f},{cam_pts[:,1].max():.1f}]")
