"""Quick packed vs unpacked comparison."""
import torch, gsplat, os, sys, math
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
torch.set_grad_enabled(False)

scene = load_ply(os.path.join(repo, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras = load_cameras_from_json(os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)
cameras = resize_cameras(cameras, 1920, 1080)
cam = cameras[0]
W, H = 1920, 1080

means3d = scene["xyz"].contiguous()
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales_exp = scene["scales"].exp().contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
viewmats = cam.world_view_transform.unsqueeze(0).contiguous()
Ks = cam.K.unsqueeze(0).contiguous()

for packed in [True, False]:
    bg = torch.zeros(3, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    out, alpha, meta = gsplat.rasterization(
        means=means3d, quats=quats, scales=scales_exp, opacities=opacities, colors=shs,
        viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=packed, tile_size=16,
        backgrounds=bg,
        render_mode="RGB", sparse_grad=False, absgrad=False, rasterize_mode="classic")
    torch.cuda.synchronize()
    n_isect = meta["isect_ids"].shape[0]
    radii_data = meta["radii"]
    if packed:
        n_radii = (radii_data > 0).sum().item() if radii_data.dim() == 1 else "unpacked_radii"
    else:
        n_radii = int((radii_data.squeeze() > 0).sum().item())
    print(f"  packed={packed}: isect_ids={n_isect:,}  radii_positive={n_radii}")
