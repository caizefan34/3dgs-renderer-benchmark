#!/usr/bin/env python3
"""PSC-P0 smoke test: validate render path, Jacobian method, SH on EPIC-05."""
import sys, os, math, importlib.util, hashlib
import numpy as np
import torch
import torch.nn.functional as F

REPO = "/root/3dgs-renderer-benchmark"
HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_V1_DIR = os.path.join(REPO, "baseline", "reference_v1")
PHASE7_DIR = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC_DIR = os.path.join(REPO, "src")
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT = "/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def load_ext_module(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

def bootstrap_c0():
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
    return core, exp

bootstrap_c0()
from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels, spherical_harmonics
from gsplat.cuda._math import _quat_scale_to_covar_preci
print("IMPORTS_OK", flush=True)

config = ReferenceV1Config()
config.scene = "room"
config.repo_root = REPO
dataset = GTDataset(scene="room", repo_root=REPO, resolution="1080p", device="cuda", background="black")
print("N_CAMS", len(dataset), flush=True)

state = torch.load(CKPT, map_location="cpu", weights_only=False)
model = GaussianModel(max_sh_degree=3)
model.restore(state, config)
N = model._xyz.shape[0]
print("N", N, "active_sh", model.active_sh_degree, flush=True)

class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2
    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                   ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
        return 1.0 - ssim_map.mean()
ssim_fn = SepSSIM(device="cuda")

cam, gt = dataset.get_item(19)
W, H = int(cam.image_width), int(cam.image_height)
print("CAM19", W, H, "center", cam.camera_center[:3], flush=True)

means = model.get_xyz
quats = model.get_rotation
scales = model.get_scaling
opac = model.get_opacity
shs = model.get_features
print("shapes", means.shape, quats.shape, scales.shape, opac.shape, shs.shape, flush=True)

viewmats = cam.viewmatrix.unsqueeze(0)
Ks = cam.K.unsqueeze(0)

# --- my manual render path ---
radii, means2d, depths, conics, comp = fully_fused_projection(
    means, None, quats, scales, viewmats, Ks, W, H,
    eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
    packed=False, calc_compensations=False)
print("proj", radii.shape, means2d.shape, conics.shape, flush=True)

dirs = (means - cam.camera_center)
masks = (radii > 0).all(dim=-1)[0]
colors = spherical_harmonics(3, dirs, shs, masks=masks)
colors = (colors + 0.5)
colors = torch.clamp_min(colors, 0.0)
print("colors", colors.shape, flush=True)

tile = 16
tw = math.ceil(W / tile); th = math.ceil(H / tile)
opac_b = opac.unsqueeze(0).expand(1, -1).contiguous()
tpg, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile, tw, th,
    packed=False, n_images=1, conics=conics, opacities=opac_b)
isect_offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
render, alphas = rasterize_to_pixels(means2d, conics, colors, opac_b, W, H, tile,
    isect_offsets, flatten_ids, backgrounds=None, packed=False, absgrad=True)
img_mine = render[0].clamp(0, 1)
print("render", img_mine.shape, float(img_mine.mean()), flush=True)

# --- trainer HIGS path ---
out = rasterize_gaussian_higs_dynamic(
    means, quats, scales, opac, shs, sh_degree=3,
    viewmats=viewmats, Ks=Ks, width=W, height=H,
    backward_mode="higs_native", enable_culling=True,
    camera_model="pinhole", render_mode="RGB", eps2d=0.3)
img_ref = out["frame"].reshape(H, W, 3).clamp(0, 1)
print("higs", img_ref.shape, float(img_ref.mean()), flush=True)
diff = (img_mine - img_ref).abs()
print("RENDER_DIFF mean=%.3e max=%.3e" % (float(diff.mean()), float(diff.max())), flush=True)

# --- loss ---
L1 = F.l1_loss(img_mine, gt)
dssim = ssim_fn(img_mine, gt)
loss = 0.8 * L1 + 0.2 * dssim
print("LOSS", float(L1), float(dssim), float(loss), flush=True)

# --- Jacobian VJP test on cohort of 64 ---
M = 64
gidx = torch.arange(M, device="cuda")
tm = means[gidx].detach().clone().requires_grad_(True)
tq = quats[gidx].detach().clone().requires_grad_(True)
ts = scales[gidx].detach().clone().requires_grad_(True)
theta = torch.cat([tm, tq, ts], dim=1).contiguous()
tv = viewmats.expand(M, 4, 4).contiguous()
tk = Ks.expand(M, 3, 3).contiguous()
# but projection is per-image; use single viewmat
tv1 = viewmats; tk1 = Ks
tm2 = tm.unsqueeze(0); tq2 = tq.unsqueeze(0); ts2 = ts.unsqueeze(0)
radii2, m2d2, dep2, con2, _ = fully_fused_projection(tm2, None, tq2, ts2, tv1, tk1, W, H,
    eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False, calc_compensations=False)
m2d2 = m2d2[0]; con2 = con2[0]
# z = (mu_2d, vech(Sigma_2d)); Sigma_2d = inv(conic)
conic_mat = torch.zeros(M, 2, 2, device="cuda")
conic_mat[:, 0, 0] = con2[:, 0]; conic_mat[:, 0, 1] = con2[:, 1]
conic_mat[:, 1, 0] = con2[:, 1]; conic_mat[:, 1, 1] = con2[:, 2]
Sigma2d = torch.linalg.inv(conic_mat)
z = torch.cat([m2d2, Sigma2d[:, 0, 0:1], Sigma2d[:, 0,1:2], Sigma2d[:,1,2:3]], dim=1)
print("z", z.shape, flush=True)
J = torch.zeros(M, 5, 10, device="cuda")
for k in range(5):
    (gk,) = torch.autograd.grad(z[:, k].sum(), theta, retain_graph=True)
    J[:, k, :] = gk
print("J", J.shape, float(J.abs().mean()), flush=True)

# compare to per-gaussian functional jacobian for gaussian 0
from torch.autograd.functional import jacobian
def z_of_theta(th):
    mm = th[0:3].unsqueeze(0); qq = th[3:7].unsqueeze(0); ss = th[7:10].unsqueeze(0)
    r, mm2, dd, cc, _ = fully_fused_projection(mm, None, qq, ss, tv1, tk1, W, H,
        eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False, calc_compensations=False)
    mm2 = mm2[0]; cc = cc[0]
    cm = torch.zeros(2,2, device="cuda"); cm[0,0]=cc[0]; cm[0,1]=cc[1]; cm[1,0]=cc[1]; cm[1,1]=cc[2]
    S = torch.linalg.inv(cm)
    return torch.cat([mm2, S[0,0:1], S[0,1:2], S[1,2:3]])
th0 = theta[0].detach().clone().requires_grad_(True)
J0 = jacobian(z_of_theta, th0)  # [5,10]
print("J0", J0.shape, "diff", float((J0 - J[0]).abs().max()), flush=True)
print("SMOKE_DONE", flush=True)
