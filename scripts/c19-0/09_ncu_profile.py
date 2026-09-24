"""
C19-0 Step 5: Nsight Compute profiling harness.

Run with: ncu --set full -o profile_rasterize python3 scripts/c19-0/09_ncu_profile.py

This script does one forward pass to trigger all kernels, then sleeps.
"""
import torch, gsplat, os, sys, math, time
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"

scene = load_ply(os.path.join(repo, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras = load_cameras_from_json(os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)
cameras = resize_cameras(cameras, 1920, 1080)
cam = cameras[0]
W, H = 1920, 1080

means3d = scene["xyz"].contiguous()
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales = scene["scales"].exp().contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
viewmats = cam.world_view_transform.unsqueeze(0).contiguous()
Ks = cam.K.unsqueeze(0).contiguous()
bg = torch.zeros(1, 3, device=DEVICE)

# Warmup
for _ in range(3):
    _ = gsplat.rasterization(means=means3d, quats=quats, scales=scales,
        opacities=opacities, colors=shs, viewmats=viewmats, Ks=Ks,
        width=W, height=H, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")

# Profiled call
out, alpha, meta = gsplat.rasterization(means=means3d, quats=quats, scales=scales,
    opacities=opacities, colors=shs, viewmats=viewmats, Ks=Ks,
    width=W, height=H, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
    backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
    rasterize_mode="classic")

torch.cuda.synchronize()
print("Rendering complete. Kernel names:")
for k, v in meta.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)}")

# Sleep to allow ncu to finalize
time.sleep(2)
