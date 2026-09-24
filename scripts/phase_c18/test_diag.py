"""Quick diagnostic: run 5 training steps and measure timing of each segment."""
import sys, time, torch
sys.path.insert(0, ".")
sys.path.insert(0, "src")

from gsplat import rasterization
from benchmark_framework import load_ply

# Inline minimal dataset
class GTDataset:
    def __init__(self, scene, repo_root, resolution="1080p", device="cuda"):
        from benchmark_framework import load_cameras_from_json, resize_cameras
        from PIL import Image
        import numpy as np
        self.repo_root = repo_root
        self.device = device
        RES = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
        tw, th = RES[resolution]
        cpath = f"{repo_root}/data/official/mipnerf360/{scene}/cameras.json"
        cam = load_cameras_from_json(cpath, device="cpu")
        cam = resize_cameras(cam, tw, th)
        self.gt_dir = f"{repo_root}/data/datasets/mipnerf360/{scene}/images"
        import pathlib
        self._valid_cameras = []
        self._valid_paths = []
        for c in cam:
            nm = getattr(c, "image_name", None) or getattr(c, "img_name", None)
            p = pathlib.Path(f"{self.gt_dir}/{pathlib.Path(nm).name}")
            if p.exists():
                self._valid_cameras.append(c)
                self._valid_paths.append(p)
        print(f"  [DS] {len(self._valid_cameras)} cameras")
        from PIL import Image
        import numpy as np
        self._gt = []
        for p in self._valid_paths:
            with Image.open(p) as s:
                if s.width != tw or s.height != th:
                    s = s.resize((tw, th), Image.LANCZOS)
                self._gt.append(torch.from_numpy(np.array(s.convert("RGBA"), dtype=np.uint8)))
        print(f"  [DS] {len(self._gt)} images loaded")

    def get_camera(self, idx):
        cam = self._valid_cameras[idx]
        for a in ["viewmatrix", "projmatrix", "camera_center", "world_view_transform", "full_proj_transform", "K"]:
            t = getattr(cam, a)
            if isinstance(t, torch.Tensor):
                setattr(cam, a, t.to(self.device))
        return cam

    def get_gt(self, idx):
        rgba = self._gt[idx]
        return rgba.to(self.device, non_blocking=True, dtype=torch.float32)[..., :3].div_(255.0).contiguous()

    def get_item(self, idx):
        return self.get_camera(idx), self.get_gt(idx)

    def __len__(self):
        return len(self._valid_cameras)

print("Loading dataset...")
ds = GTDataset("room", "/tmp/3dgs-renderer-benchmark", device="cuda")
print(f"Dataset: {len(ds)} cameras")

print("Loading PLY...")
sfm = load_ply("/tmp/3dgs-renderer-benchmark/data/official/mipnerf360/room/point_cloud.ply", device="cuda")
print(f"SfM: {sfm['xyz'].shape[0]:,}")

cam0, gt0 = ds.get_item(0)
print(f"Cam0: {cam0.image_width}x{cam0.image_height}")

# Fast forward through 5 steps
N = sfm["xyz"].shape[0]
xyz = torch.nn.Parameter(sfm["xyz"].contiguous())
op = torch.nn.Parameter(torch.logit(torch.full((N, 1), 0.1, device="cuda")))
sc = torch.nn.Parameter(sfm["scales"].contiguous())
rot = torch.nn.Parameter(sfm["rotations"].contiguous())
shs = torch.nn.Parameter(sfm["shs"].contiguous())

opt = torch.optim.Adam([
    {"params": [xyz], "lr": 1.6e-4 * float(sfm["xyz"].norm(dim=-1).max().item())},
    {"params": [op], "lr": 5e-2},
    {"params": [sc], "lr": 5e-3},
    {"params": [rot], "lr": 1e-3},
    {"params": [shs], "lr": 2.5e-3},
], eps=1e-15)

def loss_fn(img, gt, lam=0.2):
    l1 = torch.abs(img - gt).mean()
    C1, C2 = 0.01**2, 0.03**2
    mu1 = img.mean(dim=(0,1), keepdim=True)
    mu2 = gt.mean(dim=(0,1), keepdim=True)
    s1 = ((img-mu1)**2).mean(dim=(0,1), keepdim=True)
    s2 = ((gt-mu2)**2).mean(dim=(0,1), keepdim=True)
    s12 = ((img-mu1)*(gt-mu2)).mean(dim=(0,1), keepdim=True)
    ssim = (2*mu1*mu2+C1)*(2*s12+C2)/(mu1**2+mu2**2+C1)/(s1+s2+C2)
    ds = (1-ssim.mean())*0.5
    return l1*(1-lam) + ds*lam

print("Running 5 training steps with timing...")
for step in range(5):
    cam, gt = ds.get_item(step % len(ds))
    
    t0 = time.time()
    # Forward model - manual
    sx = torch.exp(sc)
    rotn = torch.nn.functional.normalize(rot, dim=-1)
    opa = torch.sigmoid(op).squeeze(-1)
    t1 = time.time()
    
    img, _, info = rasterization(
        means=xyz, quats=rotn, scales=sx, opacities=opa, colors=shs,
        viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=True, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    )
    t2 = time.time()
    
    loss = loss_fn(img[0].clamp(0,1), gt)
    t3 = time.time()
    
    opt.zero_grad(set_to_none=True)
    loss.backward()
    t4 = time.time()
    
    opt.step()
    t5 = time.time()
    
    print(f"  step {step}: forward_model={t1-t0:.3f}s raster={t2-t1:.3f}s loss={t3-t2:.3f}s backward={t4-t3:.3f}s step={t5-t4:.3f}s isects={info['isect_offsets'][0].shape}")

print("Done!")
