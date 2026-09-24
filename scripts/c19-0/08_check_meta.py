"""Check what the meta dict from gsplat.rasterization() contains."""
import torch, gsplat, json, os, sys, math
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

render_colors, render_alphas, meta = gsplat.rasterization(
    means=means3d, quats=quats, scales=scales, opacities=opacities, colors=shs,
    viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
    backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
    rasterize_mode="classic",
)

print("meta keys:", list(meta.keys()))
print()
for k, v in meta.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
    elif isinstance(v, list):
        print(f"  {k}: list of {len(v)} items")
    else:
        print(f"  {k}: {v}")

# Check if fully_fused_projection packed vs non-packed changes behavior
print("\n=== Check fully_fused_projection packed=True ===")
pj = gsplat.fully_fused_projection(
    means=means3d, covars=None, quats=quats, scales=scales,
    viewmats=viewmats, Ks=Ks, width=W, height=H,
    eps2d=0.3, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, packed=True, sparse_grad=False,
    calc_compensations=False, camera_model="pinhole",
    opacities=opacities,
)
for i, (name, t) in enumerate(zip(["means2d","covars","radii","conics","comp"], pj)):
    print(f"  {name}: shape={tuple(t.shape)}")

print("\n=== Check fully_fused_projection packed=False ===")
pj2 = gsplat.fully_fused_projection(
    means=means3d, covars=None, quats=quats, scales=scales,
    viewmats=viewmats, Ks=Ks, width=W, height=H,
    eps2d=0.3, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, packed=False, sparse_grad=False,
    calc_compensations=False, camera_model="pinhole",
    opacities=opacities,
)
for i, (name, t) in enumerate(zip(["means2d","covars","radii","conics","comp"], pj2)):
    print(f"  {name}: shape={tuple(t.shape)}")

# Check the internal data from meta
print("\n=== Meta internal data ===")
for key in ["means2d", "covars", "radii", "conics", "colors_precomp",
            "isect_ids", "flatten_ids", "isect_offsets", "tiles_per_gauss"]:
    if key in meta:
        v = meta[key]
        if isinstance(v, torch.Tensor):
            print(f"  {key}: shape={tuple(v.shape)}")
        else:
            print(f"  {key}: {type(v)}")
