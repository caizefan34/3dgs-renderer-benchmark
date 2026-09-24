#!/usr/bin/env python
"""
Training Workset Stability Audit — Phase A100.

Uses the existing diff-gaussian-rasterization CUDA kernel + pure-Python
training loop to capture workset data at each iteration.

Strictly NO CUDA kernel / renderer / optimizer / densification changes.
NO cache or reuse implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional

# ═══ Only diff-gaussian-rasterization (already compiled) ═══
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

torch.backends.cudnn.deterministic = True


# ═══════════════════════════════════════════════════════════════════
#  Helper: render from camera, return workset data
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def render_and_extract(gparams, cam, width, height, tile_size, device):
    """Render from camera viewpoint and extract worksets."""
    w2c = torch.from_numpy(cam["w2c"]).float().to(device)
    full_proj = torch.from_numpy(cam["full_proj"]).float().to(device)
    cam_center = torch.from_numpy(cam["cam_center"]).float().to(device)

    bg = torch.zeros(3, dtype=torch.float32, device=device)
    raster_settings = GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=bg, scale_modifier=1.0,
        viewmatrix=w2c, projmatrix=full_proj,
        sh_degree=0, campos=cam_center,
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    fwd = gparams.forward_params()
    means2d = torch.zeros(len(fwd["means"]), 3, device=device)

    rendered_color, rendered_radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=fwd["means"],
        means2D=means2d,
        opacities=fwd["opacities"],
        shs=fwd["shs"],
        scales=fwd["scales"],
        rotations=fwd["quats"],
        colors_precomp=None,
        cov3D_precomp=None,
    )

    # Workset A: Gaussians that rasterizer actually touches (radii > 0)
    contrib_mask = rendered_radii > 0
    visible_ids = set(torch.where(contrib_mask)[0].cpu().numpy())

    # Workset B: Active tiles from rendered alpha
    alpha_map = rendered_alpha
    if alpha_map.dim() == 3:
        alpha_map = alpha_map[0]
    n_tiles_x = math.ceil(width / tile_size)
    n_tiles_y = math.ceil(height / tile_size)
    active_tiles = set()
    for ty in range(n_tiles_y):
        y_start = ty * tile_size
        y_end = min(y_start + tile_size, height)
        for tx in range(n_tiles_x):
            x_start = tx * tile_size
            x_end = min(x_start + tile_size, width)
            if alpha_map[y_start:y_end, x_start:x_end].sum() > 0.0:
                active_tiles.add(tx + ty * n_tiles_x)

    return visible_ids, active_tiles


# ═══════════════════════════════════════════════════════════════════
#  Dataset loader
# ═══════════════════════════════════════════════════════════════════

def load_image_dataset(data_dir: Path, image_subdir: str = "images_4"):
    """Load images from COLMAP dataset with synthetic camera poses."""
    img_dir = data_dir / image_subdir
    if not img_dir.is_dir():
        img_dir = data_dir / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"No images directory in {data_dir}")

    # Find SfM points for scene bounds
    sparse_dir = data_dir / "sparse"
    pts_path = _find_pts3d(sparse_dir)
    if pts_path:
        xyz, _ = _load_points3d(pts_path)
        scene_center = xyz.mean(axis=0)
        scene_radius = float(np.max(np.linalg.norm(xyz - scene_center, axis=1)))
    else:
        scene_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scene_radius = 10.0

    img_paths = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    sample = Image.open(img_paths[0])
    width, height = sample.size
    sample.close()

    print(f"  {len(img_paths)} images, {width}x{height}, scene radius {scene_radius:.2f}")

    cameras = []
    for i, img_path in enumerate(img_paths):
        theta = 2.0 * math.pi * i / len(img_paths)
        phi = math.pi / 4.0 + 0.2 * math.sin(3.0 * theta)
        cam_dist = scene_radius * 2.5

        cam_pos = np.array([
            cam_dist * math.sin(theta) * math.cos(phi),
            cam_dist * math.sin(phi),
            cam_dist * math.cos(theta) * math.cos(phi),
        ], dtype=np.float32) + scene_center

        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        z_axis = (scene_center - cam_pos).astype(np.float32)
        z_axis /= np.linalg.norm(z_axis) + 1e-10
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-10
        y_axis = np.cross(z_axis, x_axis)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = np.vstack([x_axis, y_axis, z_axis]).T
        w2c[:3, 3] = cam_pos

        fx = width / (2.0 * math.tan(math.radians(30.0)))
        fy = height / (2.0 * math.tan(math.radians(30.0)))
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0

        tanfovx = width / (2.0 * fx)
        tanfovy = height / (2.0 * fy)

        # Original 3DGS OpenGL projection matrix (row-major)
        znear, zfar = 0.01, 100.0
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 1.0 / tanfovx
        P[1, 1] = 1.0 / tanfovy
        P[2, 2] = zfar / (zfar - znear)  # z_sign=1
        P[3, 2] = 1.0  # this is P[3,2] which encodes -z after projection
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        full_proj = P @ w2c  # row-major: P @ w2c

        cameras.append({
            "w2c": w2c,
            "full_proj": full_proj,
            "K": K,
            "width": width,
            "height": height,
            "tanfovx": tanfovx,
            "tanfovy": tanfovy,
            "cam_center": cam_pos,
            "img_path": str(img_path),
        })

    return cameras, scene_center, scene_radius


def _find_pts3d(sparse_dir: Path):
    if not sparse_dir.is_dir():
        return None
    for candidate in [sparse_dir / "points3D.bin"]:
        if candidate.is_file():
            return candidate
    for d in sorted(sparse_dir.iterdir()):
        if d.is_dir():
            c = d / "points3D.bin"
            if c.is_file():
                return c
    return None


def _load_points3d(path: Path):
    import struct
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    n_points = struct.unpack("Q", data[pos:pos+8])[0]
    pos += 8
    xyz_list, rgb_list = [], []
    for _ in range(n_points):
        _id = struct.unpack("Q", data[pos:pos+8])[0]
        pos += 8
        x, y, z = struct.unpack("3d", data[pos:pos+24]); pos += 24
        r, g, b = struct.unpack("3B", data[pos:pos+3]); pos += 3
        _ = struct.unpack("d", data[pos:pos+8])[0]; pos += 8
        track_len = struct.unpack("Q", data[pos:pos+8])[0]; pos += 8
        pos += track_len * 8
        xyz_list.append([x, y, z])
        rgb_list.append([r, g, b])
    return np.array(xyz_list, dtype=np.float32), np.array(rgb_list, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════
#  Gaussian parameter management (pure Python)
# ═══════════════════════════════════════════════════════════════════

class GaussianParams:
    """Manages Gaussian parameters as flat tensors with topology ops."""

    def __init__(self, xyz, rgb, scene_scale, device):
        N = len(xyz)
        self.device = device
        self.scene_scale = scene_scale

        self.means = torch.nn.Parameter(
            torch.from_numpy(xyz).float().to(device))
        self.opacities = torch.nn.Parameter(
            torch.zeros(N, 1, device=device).float())
        with torch.no_grad():
            self.opacities.data = torch.logit(torch.full((N, 1), 0.1, device=device))
        self.scales = torch.nn.Parameter(
            torch.log(torch.full((N, 3), 0.01, device=device)))
        self.quats = torch.nn.Parameter(
            torch.zeros(N, 4, device=device))
        self.quats.data[:, 0] = 1.0

        rgb_t = torch.from_numpy(rgb).float().to(device) / 255.0 - 0.5
        self.shs = torch.nn.Parameter(
            torch.zeros(N, 3, 3, device=device))
        self.shs.data[:, 0:1, :] = rgb_t.unsqueeze(1)

        self.optimizers = {}
        lr_config = {
            "means": 1.6e-4 * scene_scale,
            "opacities": 0.05,
            "scales": 0.005,
            "quats": 0.001,
            "shs": 0.0025,
        }
        for name, lr in lr_config.items():
            self.optimizers[name] = torch.optim.Adam(
                [getattr(self, name)], lr=lr)

        # Densification state
        self.visible_gradients = {}  # step -> (gradient norms for visible GS)
        self.max_radii2d = torch.zeros(N, device=device)

    @property
    def N(self):
        return len(self.means)

    def forward_params(self):
        """Return dict for rasterizer forward."""
        scales_act = torch.exp(self.scales).contiguous()
        quats_norm = F.normalize(self.quats, dim=-1).contiguous()
        opac_act = torch.sigmoid(self.opacities).contiguous()
        return {
            "means": self.means.contiguous(),
            "scales": scales_act,
            "quats": quats_norm,
            "opacities": opac_act,
            "shs": self.shs.contiguous(),
        }

    def densify(self, step, visible_mask, grads2d, radii):
        """Clone/split visible GS with high grad, prune low-opacity ones.
        Follows original 3DGS paper logic.
        """
        if step < 100 or step > 15000:
            return 0, 0
        if step % 100 != 0:
            return 0, 0

        n_before = self.N
        scene_scale = self.scene_scale
        n_cloned = 0

        with torch.no_grad():
            # Grad-based densification
            if visible_mask and grads2d is not None:
                # Get gradients for visible Gaussians
                grads_norm = grads2d.norm(dim=-1)  # [N]
                grad_thresh = 0.0002

                # Find GS with large 2D gradient
                high_grad_mask = grads_norm > grad_thresh

                # Split large ones, clone small ones
                scales_act = torch.exp(self.scales)
                scale_thresh = 0.01 * scene_scale
                is_small = scales_act.max(dim=-1).values < scale_thresh
                is_large = scales_act.max(dim=-1).values >= scale_thresh

                clone_mask = high_grad_mask & is_small
                split_mask = high_grad_mask & is_large
                n_cloned = int(clone_mask.sum().item())

                self._clone(clone_mask)
                self._split(split_mask)

            # Pruning
            prune_opa = 0.005
            prune_mask = torch.sigmoid(self.opacities) < prune_opa
            if step > 1000:
                # Also prune very large GS
                scales_act = torch.exp(self.scales)
                prune_mask |= scales_act.max(dim=-1).values > 0.1 * scene_scale
            self._prune(prune_mask)

        n_after = self.N
        return n_after - n_before, n_cloned

    def _clone(self, mask):
        if mask.sum() == 0:
            return
        new_params = {}
        for name in ["means", "opacities", "scales", "quats", "shs"]:
            param = getattr(self, name)
            new_p = param[mask].detach().clone()
            new_params[name] = new_p
            setattr(self, name, torch.nn.Parameter(
                torch.cat([param.detach(), new_p], dim=0)))
            self.optimizers[name].param_groups[0]["params"] = [getattr(self, name)]

        # Also clone max_radii2d
        self.max_radii2d = torch.cat([
            self.max_radii2d, self.max_radii2d[mask]], dim=0)

    def _split(self, mask):
        if mask.sum() == 0:
            return
        n_split = mask.sum().item()
        means = self.means[mask]
        scales = torch.exp(self.scales[mask])
        quats = self.quats[mask]

        # Sample splitting direction from quaternion
        rand_dir = torch.randn(n_split, 3, device=self.device)
        rand_dir = F.normalize(rand_dir, dim=-1)

        # New means offset by scale along random direction
        offsets = rand_dir * scales.max(dim=-1, keepdim=True).values * 0.5
        means_new1 = means + offsets
        means_new2 = means - offsets

        scales_new = scales / 1.6  # scale halving

        for name in ["opacities", "quats", "shs"]:
            param = getattr(self, name)
            new_p = param[mask].detach().clone()
            doubled = torch.cat([new_p, new_p], dim=0)
            setattr(self, name, torch.nn.Parameter(
                torch.cat([param.detach(), doubled], dim=0)))
            self.optimizers[name].param_groups[0]["params"] = [getattr(self, name)]

        # Means
        new_means = torch.cat([means_new1, means_new2], dim=0)
        self.means = torch.nn.Parameter(
            torch.cat([self.means.detach(), new_means], dim=0))
        self.optimizers["means"].param_groups[0]["params"] = [self.means]

        # Scales (log)
        new_scales = torch.log(
            torch.cat([scales_new, scales_new], dim=0))
        self.scales = torch.nn.Parameter(
            torch.cat([self.scales.detach(), new_scales], dim=0))
        self.optimizers["scales"].param_groups[0]["params"] = [self.scales]

        # Radii
        self.max_radii2d = torch.cat([
            self.max_radii2d,
            self.max_radii2d[mask].repeat(2),
        ], dim=0)

    def _prune(self, mask):
        if mask.sum() == 0:
            return
        keep = ~mask
        for name in ["means", "opacities", "scales", "quats", "shs"]:
            param = getattr(self, name)
            setattr(self, name, torch.nn.Parameter(param[keep].detach()))
            self.optimizers[name].param_groups[0]["params"] = [getattr(self, name)]
        self.max_radii2d = self.max_radii2d[keep]


# ═══════════════════════════════════════════════════════════════════
#  Overlap helpers
# ═══════════════════════════════════════════════════════════════════

def jaccard_overlap(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 1.0


def compute_overlap_stats(history: list[set]) -> dict:
    overlaps_by_lag = {lag: [] for lag in [1, 2, 4, 8]}
    for t in range(len(history) - 1):
        for lag in [1, 2, 4, 8]:
            if t + lag < len(history):
                overlaps_by_lag[lag].append(jaccard_overlap(history[t], history[t + lag]))
    stats = {}
    for lag, vals in overlaps_by_lag.items():
        if not vals:
            continue
        arr = np.array(vals)
        stats[f"lag_{lag}"] = {
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "min": float(float(arr.min())),
            "max": float(float(arr.max())),
        }
    return stats


def classify_stability(stats: dict) -> str:
    median = stats.get("lag_1", {}).get("p50", 0)
    if median >= 0.9:
        return "HIGH"
    elif median >= 0.7:
        return "MODERATE"
    return "LOW"


# ═══════════════════════════════════════════════════════════════════
#  Workset extraction (from diff-gaussian-rasterization)
# ═══════════════════════════════════════════════════════════════════

def extract_visible_gaussians(rasterizer, means3d, viewmatrix, projmatrix):
    """Use markVisible to get visible Gaussians from current viewpoint."""
    with torch.no_grad():
        visible = rasterizer.markVisible(means3d)
    return set(int(i) for i in torch.where(visible)[0].cpu().numpy())


def compute_active_tiles_and_membership(means3d, opacities, scales, quats,
                                         w2c, K, width, height, tile_size=16):
    """Compute which tiles each Gaussian covers via CPU/Python projection.
    
    This is a pure Python reconstruction to avoid modifying CUDA code.
    We project each Gaussian to screen space and determine tile coverage.
    """
    device = means3d.device
    with torch.no_grad():
        # World to camera
        R = w2c[:3, :3].to(device)
        t = w2c[:3, 3].to(device)

        # Project means to camera space
        means_cam = means3d @ R.T + t  # [N, 3]

        # Only front-facing Gaussians
        visible_z = means_cam[:, 2] > 0.1
        
        # Project to screen
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
        
        means2d = torch.zeros(len(means3d), 2, device=device)
        means2d[visible_z, 0] = means_cam[visible_z, 0] / means_cam[visible_z, 2] * fx + cx
        means2d[visible_z, 1] = means_cam[visible_z, 1] / means_cam[visible_z, 2] * fy + cy

        # Compute screen-space covariance (approximate 2D radius)
        quats_norm = F.normalize(quats, dim=-1)
        scales_act = torch.exp(scales)
        
        # 2D Gaussian radius estimation using Jacobian of projection
        # Using the same approach as the original 3DGS paper
        scales_2d = scales_act * fx / (means_cam[:, 2:3] + 1e-10)  # rough approximation
        radii = scales_2d.max(dim=-1).values * 3.0  # 3-sigma
        
        # Clip to image
        in_img = (means2d[:, 0] > -radii) & (means2d[:, 0] < width + radii) & \
                 (means2d[:, 1] > -radii) & (means2d[:, 1] < height + radii) & visible_z
        
        # Determine tile coverage
        n_tiles_x = math.ceil(width / tile_size)
        n_tiles_y = math.ceil(height / tile_size)
        tile_ids = {}
        active_tiles = set()
        membership = set()
        
        for gid in torch.where(in_img)[0]:
            gid = int(gid)
            cx_g, cy_g = means2d[gid, 0].item(), means2d[gid, 1].item()
            r = radii[gid].item()
            
            t_min_x = max(0, int((cx_g - r) / tile_size))
            t_max_x = min(n_tiles_x - 1, int((cx_g + r) / tile_size))
            t_min_y = max(0, int((cy_g - r) / tile_size))
            t_max_y = min(n_tiles_y - 1, int((cy_g + r) / tile_size))
            
            for tx in range(t_min_x, t_max_x + 1):
                for ty in range(t_min_y, t_max_y + 1):
                    tid = tx + ty * n_tiles_x
                    active_tiles.add(tid)
                    membership.add((gid, tid))
        
        return active_tiles, membership, means2d, radii


# ═══════════════════════════════════════════════════════════════════
#  Main training + audit
# ═══════════════════════════════════════════════════════════════════

def run_audit(args):
    device = torch.device("cuda")
    data_dir = Path(args.data_dir)
    max_steps = args.max_steps
    tile_size = args.tile_size

    print(f"Loading dataset from {data_dir} ...")
    cameras, scene_center, scene_radius = load_image_dataset(data_dir)
    n_cameras = len(cameras)
    width, height = cameras[0]["width"], cameras[0]["height"]

    # ── Initialize Gaussians ──
    sparse_dir = data_dir / "sparse"
    pts_path = _find_pts3d(sparse_dir)
    if pts_path:
        xyz, rgb = _load_points3d(pts_path)
        print(f"  {len(xyz)} initial Gaussians from SfM")
    else:
        xyz = np.random.randn(5000, 3).astype(np.float32) * scene_radius * 0.3 + scene_center
        rgb = np.ones((len(xyz), 3), dtype=np.uint8) * 128
        print(f"  {len(xyz)} random initial Gaussians")

    gparams = GaussianParams(xyz, rgb, scene_radius, device)
    print(f"  Scene scale: {scene_radius:.4f}")

    # Pre-load images
    img_tensors = []
    for cam in cameras:
        if os.path.isfile(cam["img_path"]):
            img = Image.open(cam["img_path"]).convert("RGB")
            img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).to(device)
            img_tensors.append(img_t)
        else:
            img_tensors.append(torch.ones(height, width, 3, device=device) * 0.5)
    print(f"  {len(img_tensors)} images loaded")

    # ── Recorder ──
    vg_sets = [set() for _ in range(max_steps)]       # training-view worksets
    at_sets = [set() for _ in range(max_steps)]
    mb_sets = None
    n_vis = [0] * max_steps
    n_tiles = [0] * max_steps
    n_gaussians = [0] * max_steps
    dens_events = []
    prune_events = []

    # Fixed evaluation cameras (first 5 cameras)
    eval_cameras = cameras[:5]
    eval_steps = list(range(0, max_steps, 50))  # every 50 steps
    eval_vg_sets = {}  # step -> [set per eval cam]
    eval_at_sets = {}

    # ── Training ──
    print(f"\nTraining {max_steps} steps (tile_size={tile_size}) ...")
    print(f"  Periodic eval every 50 steps from {len(eval_cameras)} fixed cameras")
    t0 = time.time()

    def ssim_loss(img1, img2):
        """Simple SSIM implementation."""
        C1, C2 = 0.01**2, 0.03**2
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        sigma1_sq = F.avg_pool2d(img1**2, 3, 1, 1) - mu1**2
        sigma2_sq = F.avg_pool2d(img2**2, 3, 1, 1) - mu2**2
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2
        num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        return 1 - num.mean() / den.mean()

    for step in range(max_steps):
        cam_idx = np.random.randint(n_cameras)
        cam = cameras[cam_idx]

        # Camera setup
        w2c = torch.from_numpy(cam["w2c"]).float().to(device)
        full_proj = torch.from_numpy(cam["full_proj"]).float().to(device)
        cam_center = torch.from_numpy(cam["cam_center"]).float().to(device)

        bg = torch.zeros(3, dtype=torch.float32, device=device)
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=cam["tanfovx"],
            tanfovy=cam["tanfovy"],
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        fwd = gparams.forward_params()
        means2d = torch.zeros(len(fwd["means"]), 3, device=device, requires_grad=True)

        rendered_color, rendered_radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=fwd["means"],
            means2D=means2d,
            opacities=fwd["opacities"],
            shs=fwd["shs"],
            scales=fwd["scales"],
            rotations=fwd["quats"],
            colors_precomp=None,
            cov3D_precomp=None,
        )

        # ── Worksets from training view ──
        contrib_mask = rendered_radii > 0
        vg_sets[step] = set(torch.where(contrib_mask)[0].cpu().numpy())
        n_vis[step] = len(vg_sets[step])
        n_gaussians[step] = gparams.N

        alpha_map = rendered_alpha
        if alpha_map.dim() == 3:
            alpha_map = alpha_map[0]
        n_tiles_x_here = math.ceil(width / tile_size)
        n_tiles_y_here = math.ceil(height / tile_size)
        active_tiles = set()
        for ty in range(n_tiles_y_here):
            y_start = ty * tile_size
            y_end = min(y_start + tile_size, height)
            for tx in range(n_tiles_x_here):
                x_start = tx * tile_size
                x_end = min(x_start + tile_size, width)
                if alpha_map[y_start:y_end, x_start:x_end].sum() > 0.0:
                    active_tiles.add(tx + ty * n_tiles_x_here)
        at_sets[step] = active_tiles
        n_tiles[step] = len(active_tiles)

        # ── Periodic same-viewpoint eval ──
        if step in eval_steps:
            evg = []
            eat = []
            for ecam in eval_cameras:
                e_vis, e_at = render_and_extract(gparams, ecam, width, height, tile_size, device)
                evg.append(e_vis)
                eat.append(e_at)
            eval_vg_sets[step] = evg
            eval_at_sets[step] = eat

        # ── Loss ──
        rendered_img = rendered_color.permute(1, 2, 0).clamp(0, 1)
        gt = img_tensors[cam_idx]
        if gt.shape[:2] != (height, width):
            gt = F.interpolate(gt.permute(2, 0, 1).unsqueeze(0),
                              size=(height, width), mode="bilinear"
                              ).squeeze(0).permute(1, 2, 0)

        l1_loss = F.l1_loss(rendered_img, gt)
        ssim_val = ssim_loss(
            rendered_img.permute(2, 0, 1).unsqueeze(0),
            gt.permute(2, 0, 1).unsqueeze(0),
        )
        loss = (1.0 - args.ssim_weight) * l1_loss + args.ssim_weight * ssim_val

        # ── Backward ──
        loss.backward()

        # ── Densification (before optimizer step) ──
        # Collect gradient info from means2d (which has requires_grad=True)
        if means2d.grad is not None:
            grads2d = means2d.grad.detach()
        else:
            grads2d = None

        n_before = gparams.N
        _, n_clone = gparams.densify(step, True, grads2d, rendered_radii)
        n_after = gparams.N

        if n_after > n_before:
            dens_events.append(step)
        elif n_after < n_before:
            prune_events.append(step)

        # ── Optimizer step ──
        for opt in gparams.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == max_steps - 1:
            el = time.time() - t0
            print(f"  step {step:5d}/{max_steps} | GS={gparams.N:6d} | "
                  f"vis={n_vis[step]:6d} | tiles={n_tiles[step]:5d} | "
                  f"loss={loss.item():.4f} | {el:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    #  Analysis
    # ═══════════════════════════════════════════════════════════════

    print("\n=== Computing statistics ...")

    def valid(sets):
        return [s for s in sets if len(s) > 0]

    vg_v = valid(vg_sets)
    at_v = valid(at_sets)

    vg_stats = compute_overlap_stats(vg_v)
    at_stats = compute_overlap_stats(at_v)

    results = {
        "workload": {
            "scene": data_dir.name,
            "n_images": n_cameras,
            "image_size": f"{width}x{height}",
            "max_steps": max_steps,
            "tile_size": tile_size,
        },
        "visible_gaussian_stability": vg_stats,
        "active_tile_stability": at_stats,
        "lag_analysis": {
            "visible_gaussian": vg_stats,
            "active_tile": at_stats,
        },
        "stability_classification": {
            "visible_gaussian": classify_stability(vg_stats),
            "active_tile": classify_stability(at_stats),
        },
        "densification_events": {
            "count": len(dens_events),
            "steps": dens_events[:50],
        },
        "pruning_events": {
            "count": len(prune_events),
            "steps": prune_events[:50],
        },
        "gaussian_count_evolution": {
            "min": int(min(n_gaussians)),
            "p50": int(np.median(n_gaussians)),
            "p90": int(np.percentile(n_gaussians, 90)),
            "max": int(max(n_gaussians)),
            "final": int(n_gaussians[-1]),
            "delta_max": int(max(np.abs(np.diff(n_gaussians)))) if len(n_gaussians) > 1 else 0,
        },
        "renderer_workload": {
            "n_visible": {
                "p50": int(np.median(n_vis)),
                "p90": int(np.percentile(n_vis, 90)),
                "max": int(max(n_vis)),
            },
            "n_tiles": {
                "p50": int(np.median(n_tiles)),
                "p90": int(np.percentile(n_tiles, 90)),
                "max": int(max(n_tiles)),
            },
        },
    }

    # Phase analysis (from visible & tile sets only)
    def phase_overlap(sets_list, start, end):
        s = [x for x in sets_list[start:end] if len(x) > 0]
        return compute_overlap_stats(s) if len(s) > 1 else {}

    phase_ranges = [("early", 0, min(100, max_steps)),
                    ("middle", 101, min(350, max_steps)),
                    ("late", 351, max_steps)]
    for pname, ps, pe in phase_ranges:
        if ps < pe:
            pa = {}
            for sn, sl in [("visible_gaussian", vg_sets),
                           ("active_tile", at_sets)]:
                ov = phase_overlap(sl, ps, pe)
                if ov:
                    pa[sn] = ov
            if pa:
                results.setdefault("phase_analysis", {})[pname] = pa

    # ── Same-viewpoint stability analysis ──
    # Compare worksets from the SAME evaluation camera across training steps
    print(f"\n  Same-viewpoint stability over {len(eval_steps)} checkpoints...")
    eval_overlap_vg_by_cam = []
    eval_overlap_at_by_cam = []
    for ci in range(len(eval_cameras)):
        vg_by_step = [eval_vg_sets[s][ci] for s in eval_steps if s in eval_vg_sets]
        at_by_step = [eval_at_sets[s][ci] for s in eval_steps if s in eval_at_sets]
        vg_ov = compute_overlap_stats(vg_by_step)
        at_ov = compute_overlap_stats(at_by_step)
        if vg_ov:
            eval_overlap_vg_by_cam.append(vg_ov.get("lag_1", {}).get("p50", 0))
        if at_ov:
            eval_overlap_at_by_cam.append(at_ov.get("lag_1", {}).get("p50", 0))

    results["same_viewpoint_stability"] = {
        "visible_gaussian_median_p50": float(np.median(eval_overlap_vg_by_cam)) if eval_overlap_vg_by_cam else -1,
        "active_tile_median_p50": float(np.median(eval_overlap_at_by_cam)) if eval_overlap_at_by_cam else -1,
        "visible_gaussian_per_camera": [float(v) for v in eval_overlap_vg_by_cam],
        "active_tile_per_camera": [float(v) for v in eval_overlap_at_by_cam],
    }

    # Update classification with same-viewpoint data
    sv_vg_median = results["same_viewpoint_stability"]["visible_gaussian_median_p50"]
    sv_at_median = results["same_viewpoint_stability"]["active_tile_median_p50"]
    
    def cls_same(v):
        return "HIGH" if v >= 0.9 else "MODERATE" if v >= 0.7 else "LOW"
    
    results["stability_classification"]["visible_gaussian_same_view"] = cls_same(sv_vg_median)
    results["stability_classification"]["active_tile_same_view"] = cls_same(sv_at_median)
    def event_analysis(event_steps, sets_list, window=2):
        if not event_steps:
            return {}
        lags = list(range(-window, window + 1))
        result = {}
        for lag in lags:
            vals = []
            for ev in event_steps[:30]:
                src, tgt = ev + lag, ev + lag + 1
                if src < 0 or tgt >= max_steps:
                    continue
                if sets_list[src] and sets_list[tgt]:
                    vals.append(jaccard_overlap(sets_list[src], sets_list[tgt]))
            if vals:
                result[f"lag_{lag:+d}"] = {
                    "visible": float(np.median(vals)),
                    "count": len(vals),
                }
        return result

    results["densification_analysis"] = event_analysis(dens_events, vg_sets)
    results["pruning_analysis"] = event_analysis(prune_events, vg_sets)

    # Overall judgment — TWO measurements:
    # 1) Training-view (random camera each step, confounded by viewpoint change)
    # 2) Same-viewpoint (fixed camera evaluated every 50 steps, measures true param-stability)
    sv_vg = results["same_viewpoint_stability"]["visible_gaussian_median_p50"]
    sv_at = results["same_viewpoint_stability"]["active_tile_median_p50"]

    sv_vg_cls = "HIGH" if sv_vg >= 0.9 else "MODERATE" if sv_vg >= 0.7 else "LOW"
    sv_at_cls = "HIGH" if sv_at >= 0.9 else "MODERATE" if sv_at >= 0.7 else "LOW"

    # Same-viewpoint is the DECISIVE metric: it measures pure parameter-evolution stability
    if sv_vg_cls == "HIGH" and sv_at_cls == "HIGH":
        opportunity = "RESEARCH OPPORTUNITY"
        temporal = f"YES (VG={sv_vg:.3f}, AT={sv_at:.3f})"
        next_q = "Design unified workset cache with tile-level rebuild"
    elif sv_vg_cls == "HIGH":
        opportunity = "PARTIAL RESEARCH OPPORTUNITY"
        temporal = f"YES for VG ({sv_vg:.3f}), NO for AT ({sv_at:.3f})"
        next_q = "Pursue Gaussian-level reuse; investigate tile instability root cause"
    elif sv_vg_cls == "MODERATE":
        opportunity = "FURTHER ANALYSIS NEEDED"
        temporal = f"Borderline (VG={sv_vg:.3f})"
        next_q = "Run full 30K training to measure convergence-phase stability"
    else:
        opportunity = "DROP CANDIDATE"
        temporal = "NO"
        next_q = "Workset instability precludes state reuse"

    results.update({
        "research_opportunity": opportunity,
        "temporal_persistence": temporal,
        "next_question": next_q,
    })

    # ── Save ──
    json_path = Path(args.output_json)
    report_path = Path(args.output_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON -> {json_path}")

    # CSV
    csv_path = json_path.parent / "training_workset_overlap_series.csv"
    with open(csv_path, "w") as f:
        f.write("iteration,visible_overlap_lag1,tile_overlap_lag1,n_gaussian\n")
        for t in range(max_steps - 1):
            ov_v = jaccard_overlap(vg_sets[t], vg_sets[t + 1]) if vg_sets[t] and vg_sets[t + 1] else -1
            ov_t = jaccard_overlap(at_sets[t], at_sets[t + 1]) if at_sets[t] and at_sets[t + 1] else -1
            f.write(f"{t},{ov_v:.6f},{ov_t:.6f},{n_gaussians[t]}\n")
    print(f"  CSV -> {csv_path}")
    # ── Report ──
    report = _gen_report(results)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report -> {report_path}")

    # ── Terminal ──
    _print_terminal(results)
    print("\nOPTIMIZATION:")
    print("NOT STARTED")


def _gen_report(results):
    lines = [f"# Training Workset Stability Audit\n"]
    wl = results["workload"]
    lines.append(f"Scene: `{wl['scene']}`  ")
    lines.append(f"Image: {wl['image_size']}, Steps: {wl['max_steps']}, Tile: {wl['tile_size']}\n")
    lines.append("---\n")

    lines.append("## Gaussian Count Evolution\n")
    gc = results["gaussian_count_evolution"]
    lines.append(f"- Min: {gc['min']}, P50: {gc['p50']}, P90: {gc['p90']}, Max: {gc['max']}, Final: {gc['final']}\n")

    lines.append("## Training-View (Confounded) Stability\n")
    lines.append("Measured across random camera views at each step. "
                 "Low overlap expected due to viewpoint changes.\n")
    for name, key in [("Visible Gaussian (A)", "visible_gaussian_stability"),
                      ("Active Tile (B)", "active_tile_stability")]:
        lines.append(f"### {name}\n")
        stats = results.get(key, {})
        for lag in [1, 2, 4, 8]:
            s = stats.get(f"lag_{lag}", {})
            if s:
                lines.append(f"- **Lag-{lag}**: P50={s['p50']:.4f}, Mean={s['mean']:.4f}, P90={s['p90']:.4f}\n")

    lines.append("## Same-Viewpoint Stability (Decisive)\n")
    lines.append("Measured from 5 fixed cameras every 50 steps. "
                 "This measures **pure parameter-evolution stability**.\n")
    sv = results.get("same_viewpoint_stability", {})
    lines.append(f"- **Visible Gaussian** median P50: **{sv.get('visible_gaussian_median_p50', -1):.3f}** "
                 f"(classification: **{results['stability_classification'].get('visible_gaussian_same_view', '?')}**)\n")
    lines.append(f"- **Active Tile** median P50: **{sv.get('active_tile_median_p50', -1):.3f}** "
                 f"(classification: **{results['stability_classification'].get('active_tile_same_view', '?')}**)\n")
    lines.append("Per-camera VG overlap at lag-1:\n")
    for i, v in enumerate(sv.get("visible_gaussian_per_camera", [])):
        lines.append(f"  - Camera {i+1}: **{v:.3f}**\n")
    lines.append("Per-camera AT overlap at lag-1:\n")
    for i, v in enumerate(sv.get("active_tile_per_camera", [])):
        lines.append(f"  - Camera {i+1}: **{v:.3f}**\n")

    sc = results.get("stability_classification", {})
    lines.append("## Classification\n")
    for k, v in sc.items():
        lines.append(f"- {k}: **{v}**\n")

    lines.append("## Overall\n")
    lines.append(f"- Temporal persistence: {results.get('temporal_persistence', '?')}")
    lines.append(f"- Research opportunity: {results.get('research_opportunity', '?')}")
    lines.append(f"- Next: {results.get('next_question', '?')}\n")
    return "\n".join(lines)


def _print_terminal(results):
    print("\n" + "=" * 50)
    print("=== Training Workset Stability Audit ===")
    wl = results["workload"]
    print(f"\nTraining: {wl['scene']}, {wl['max_steps']} steps, {wl['image_size']}")

    for name, key in [("Visible Gaussian", "visible_gaussian_stability"),
                      ("Active Tiles", "active_tile_stability")]:
        stats = results.get(key, {})
        cls = results.get("stability_classification", {}).get(
            {"Visible Gaussian": "visible_gaussian",
             "Active Tiles": "active_tile"}.get(name, ""), "?")
        print(f"\n{name}:")
        for lag in [1, 2, 4, 8]:
            s = stats.get(f"lag_{lag}", {})
            if s:
                print(f"  lag-{lag} P50={s['p50']:.4f} mean={s['mean']:.4f} P90={s['p90']:.4f} P99={s['p99']:.4f}")
        print(f"  classification = {cls}")

    gc = results.get("gaussian_count_evolution", {})
    print(f"\nGaussians: min={gc.get('min')}, P50={gc.get('p50')}, max={gc.get('max')}")
    sv = results.get("same_viewpoint_stability", {})
    print(f"\nSame-viewpoint (decisive):")
    print(f"  Visible Gaussian: {sv.get('visible_gaussian_median_p50', -1):.3f} "
          f"({results['stability_classification'].get('visible_gaussian_same_view', '?')})")
    print(f"  Active Tiles:     {sv.get('active_tile_median_p50', -1):.3f} "
          f"({results['stability_classification'].get('active_tile_same_view', '?')})")
    print(f"\nTemporal persistence: {results.get('temporal_persistence', '?')}")
    print(f"Research opportunity: {results.get('research_opportunity', '?')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--output-json", default="results/phase-a100/training_workset_stability_audit.json")
    parser.add_argument("--output-report", default="reports/phase-a100/training_workset_stability_audit.md")
    run_audit(parser.parse_args())
