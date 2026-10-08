"""Scene loading for the MomentGrad-GS oracle lane.

Reuses the repo COLMAP reader (fake_pycolmap.Reconstruction) to load MipNeRF 360
scenes and initializes a standard 3D Gaussian set (INRIA/3DGS convention).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake_pycolmap import Reconstruction  # noqa: E402


@dataclass
class Camera:
    uid: int
    K: torch.Tensor          # 3x3 intrinsics
    width: int
    height: int
    camtoworld: torch.Tensor # 4x4
    worldtocam: torch.Tensor # 4x4
    name: str


@dataclass
class GaussianInit:
    means: torch.Tensor     # [N,3] float32
    colors: torch.Tensor    # [N,3] float32 (SH DC term, linear-ish)
    scales: torch.Tensor    # [N,3] (log scale)
    quats: torch.Tensor     # [N,4] (w,x,y,z)
    opacities: torch.Tensor # [N] (logit)


def _qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _intrinsics(cam):
    p = np.array(cam.params, dtype=np.float64)
    W, H = cam.width, cam.height
    if cam.model == 0:  # SIMPLE_PINHOLE
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif cam.model == 1:  # PINHOLE
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif cam.model == 2:  # SIMPLE_RADIAL
        f, cx, cy, _ = p[0], p[1], p[2], p[3]
        fx = fy = f
    elif cam.model == 3:  # RADIAL
        fx, fy, cx, cy, _ = p[0], p[1], p[2], p[3], p[4]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, W, H


def load_scene(scene_dir: str, split: str = "train", every: int = 8,
               test_offset: int = 0, max_points: Optional[int] = None,
               device: str = "cuda"):
    """Load a MipNeRF360 scene. split in {train,test}. every/test_offset define
    the Graphdeco split (test = idx % every == test_offset)."""
    rec = Reconstruction(Path(scene_dir) / "sparse" / "0")
    img_ids = sorted(rec.images.keys())
    cams: List[Camera] = []
    for uid, iid in enumerate(img_ids):
        keep = (uid % every == test_offset) if split == "test" else (uid % every != test_offset)
        if not keep:
            continue
        img = rec.images[iid]
        K, W, H = _intrinsics(img.camera)
        R = _qvec_to_rotmat(img.qvec)   # world->cam rotation
        t = img.tvec.reshape(3)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        cams.append(Camera(
            uid=len(cams),
            K=torch.from_numpy(K).float(),
            width=W, height=H,
            camtoworld=torch.from_numpy(c2w).float(),
            worldtocam=torch.from_numpy(w2c).float(),
            name=img.name,
        ))

    # Initialize Gaussians from COLMAP points.
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=np.float32)
    rgb = np.array([p.rgb for p in pts], dtype=np.float32) / 255.0
    if max_points is not None and xyz.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(xyz.shape[0], max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    n = xyz.shape[0]
    # Dist2-based per-point scale (INRIA heuristic, simplified).
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    _, dists = tree.query(xyz, k=2)
    dist2 = dists[:, 1]
    scale = np.clip(dist2 * 0.5, 1e-4, 0.02)  # world std cap -> moderate screen footprints
    scales = np.tile(np.log(scale)[:, None], (1, 3)).astype(np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.log(0.1 / (1 - 0.1) * np.ones((n, 1), dtype=np.float32)).squeeze(-1)
    # SH DC term from color: C = 0.5 + 0.28209479177387814 * SH_C  =>  SH_C = (C-0.5)/0.28209...
    sh_c0 = 0.28209479177387814
    colors_sh = (rgb - 0.5) / sh_c0
    colors_sh = np.clip(colors_sh, -4, 4).astype(np.float32)

    init = GaussianInit(
        means=torch.from_numpy(xyz).to(device),
        colors=torch.from_numpy(colors_sh).to(device),
        scales=torch.from_numpy(scales).to(device),
        quats=torch.from_numpy(quats).to(device),
        opacities=torch.from_numpy(opacities).to(device),
    )
    return cams, init


def pick_views(cams: List[Camera], k: int = 4, seed: int = 0) -> List[int]:
    """Deterministically pick k representative view indices (spread across list)."""
    rng = np.random.default_rng(seed)
    if len(cams) <= k:
        return list(range(len(cams)))
    return sorted(rng.choice(len(cams), k, replace=False).tolist())
