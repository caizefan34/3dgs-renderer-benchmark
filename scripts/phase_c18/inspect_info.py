import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")
import torch
from gsplat import rasterization
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
root = "."
scene = "room"
sfm = load_ply(f"{root}/data/official/mipnerf360/{scene}/point_cloud.ply", device="cuda")
cams = resize_cameras(load_cameras_from_json(f"{root}/data/official/mipnerf360/{scene}/cameras.json", device="cpu"), 1920, 1080)
cam = cams[0]
cam.viewmatrix = cam.viewmatrix.cuda(); cam.K = cam.K.cuda()
N = sfm["xyz"].shape[0]
_, _, info = rasterization(means=sfm["xyz"], quats=torch.nn.functional.normalize(sfm["rotations"], dim=-1), scales=torch.exp(sfm["scales"]), opacities=torch.full((N,), 0.1, device="cuda"), colors=sfm["shs"], viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height, tile_size=16, packed=True, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
for key, value in info.items():
    if isinstance(value, torch.Tensor): print(key, tuple(value.shape), value.dtype)
    else: print(key, type(value).__name__, value)
