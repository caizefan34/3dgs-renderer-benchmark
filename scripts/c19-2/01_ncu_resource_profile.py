"""
C19-2 Goal 1: ncu resource profiling harness.

This script runs a single forward pass and holds CUDA long enough for
ncu to capture. Run it via:

  ncu --section-folder /usr/lib/nsight-compute/sections ... --kernel-name rasterize_to_pixels_3dgs_fwd_kernel ... python3 01_ncu_resource_profile.py

The actual ncu invocation is handled by the bash wrapper 01_ncu_resource_profile.sh.
"""
import torch, gsplat, os, sys, math, time, argparse

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

parser = argparse.ArgumentParser()
parser.add_argument("--tile-size", type=int, default=16)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

DEVICE = f"cuda:{args.gpu}"
torch.set_grad_enabled(False)

scene = load_ply(os.path.join(repo, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras = load_cameras_from_json(os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)
cameras = resize_cameras(cameras, 1920, 1080)
cam = cameras[0]
W, H = 1920, 1080

means3d = scene["xyz"].contiguous().to(DEVICE)
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous().to(DEVICE)
scales = scene["scales"].exp().contiguous().to(DEVICE)
opacities = torch.sigmoid(scene["opacity"]).contiguous().to(DEVICE)
shs = scene["shs"].contiguous().to(DEVICE)
viewmats = cam.world_view_transform.unsqueeze(0).contiguous().to(DEVICE)
Ks = cam.K.unsqueeze(0).contiguous().to(DEVICE)
bg = torch.zeros(1, 3, device=DEVICE)

# Warmup 3× to trigger JIT
for _ in range(3):
    out, alpha, meta = gsplat.rasterization(
        means=means3d, quats=quats, scales=scales, opacities=opacities, colors=shs,
        viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=args.tile_size,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
torch.cuda.synchronize()

print(f"Warmup complete. Running profiled pass (tile_size={args.tile_size})...")

# Profiled call
out, alpha, meta = gsplat.rasterization(
    means=means3d, quats=quats, scales=scales, opacities=opacities, colors=shs,
    viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=args.tile_size,
    backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
    rasterize_mode="classic")
torch.cuda.synchronize()

n_isect = meta["isect_ids"].shape[0]
print(f"Intersections: {n_isect:,}")
print(f"Kernel names:")
for k, v in meta.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)}")

print("Profiled pass complete. Holding for ncu...")
time.sleep(3)
