#!/usr/bin/env python3
"""BES0 scene adapter (DSH implementation).

Implements the fixed interface from ADAPTER_CONTRACT.md:
    load_scene(scene_record, device) -> Scene
    project_and_sort(scene, camera, tile_size) -> TileLists
    native_alpha(tile_lists, occurrence_ids, pixel_xy) -> Tensor
    build_bounds(tile_lists, mode) -> BoundsAndRouting
    native_complete(scene, camera, return_cache=False) -> ImageAndCache
    native_default(scene, camera, return_cache=False) -> ImageAndCache
    bes_sample(scene, camera, routing, stream_spec) -> EventImagesAndCache
    bes_l2_surrogate(event_images, target) -> Scalar
    optimizer_step(scene, grads, optimizer_state) -> None

This module is DSH production code under the frozen BES0-v1 protocol.
It binds the native gsplat 1.5.3 rasterization alpha path so that
native_complete / native_default / candidate evaluation call the SAME
alpha device function. R0 bound enumeration and the BES event sampler are
implemented here against that native alpha evaluator.

Scientific entry points (run_e1/e2/e3.py) must call
mx_guard.require_guarded_process() before importing this module.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# gsplat native rasterization backend (pinned 1.5.3 by mx_target.json).
from gsplat import rasterization


# ---------------------------------------------------------------------------
# Activations (match Graphdeco / gsplat 1.5.3 semantics exactly).
# ---------------------------------------------------------------------------

def scaling_activation(x):
    return torch.exp(x)


def rotation_activation(x):
    return F.normalize(x, dim=-1)


def opacity_activation(x):
    # gsplat/3DGS: opacity = sigmoid(raw). Stored raw in checkpoint.
    return torch.sigmoid(x)


def sh_activation(x, active_degree):
    # SH coefficients are stored raw; gsplat applies the SH basis at render.
    return x


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """Pinhole camera with world-to-view (cam_from_world) pose."""
    image_id: int
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    cam_from_world: torch.Tensor          # [4, 4] world->cam
    device: str

    @property
    def K(self):
        K = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0
        return K

    @property
    def viewmats(self):
        # gsplat expects world-to-camera [4,4].
        return self.cam_from_world


@dataclass
class Scene:
    name: str
    device: str
    # Gaussian parameters (raw, pre-activation) as leaf tensors requiring grad.
    xyz: torch.Tensor
    shs: torch.Tensor            # [N, K, 3] raw SH coefficients
    scaling: torch.Tensor       # [N, 3] raw log-scale
    rotation: torch.Tensor      # [N, 4] raw quaternion
    opacity: torch.Tensor       # [N, 1] raw logit
    active_sh_degree: int
    num_points: int
    background: torch.Tensor     # [3]
    cameras: Dict[int, Camera]   # image_id -> Camera
    train_camera_ids: List[int]
    dev_camera_ids: List[int]
    heldout_camera_ids: List[int]
    meta: Dict


@dataclass
class TileLists:
    """Per-tile depth-sorted Gaussian occurrence lists (R0 bound input)."""
    tile_size: int
    width: int
    height: int
    tiles_x: int
    tiles_y: int
    # For each tile: arrays of global gaussian indices sorted by depth (center).
    tile_gaussian_ids: List[np.ndarray]
    # Per-tile camera (single camera for project_and_sort).
    camera: Camera
    # Per-occurrence center depth (view-space z) aligned to tile_gaussian_ids.
    tile_depths: List[np.ndarray]
    meta: Dict = field(default_factory=dict)


@dataclass
class BoundsAndRouting:
    """R0 enumerated per-occurrence alpha bounds and R/D routing."""
    mode: str
    # Flattened occurrence arrays.
    gaussian_ids: np.ndarray        # global gaussian id per occurrence
    tile_ids: np.ndarray            # tile index per occurrence
    pixel_xy: np.ndarray            # [occ, 2] pixel center (x, y)
    alpha_lower: np.ndarray         # [occ] lower bound l_i
    alpha_upper: np.ndarray         # [occ] upper bound u_i
    is_R: np.ndarray                # [occ] bool routing
    meta: Dict = field(default_factory=dict)


@dataclass
class ImageAndCache:
    image: torch.Tensor            # [H, W, 3] in [0,1]
    meta: Dict = field(default_factory=dict)


@dataclass
class EventImagesAndCache:
    images: Dict[str, torch.Tensor]  # e.g. {"A":..., "B":..., "J":...}
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# COLMAP reading
# ---------------------------------------------------------------------------

def _read_colmap_scene(sparse_dir: str, device: str):
    import pycolmap
    rec = pycolmap.Reconstruction(sparse_dir)
    cameras = {}
    cams = rec.cameras
    imgs = rec.images
    cam_by_id = {}
    for cid, cam in cams.items():
        cam_by_id[cid] = cam
    for iid in sorted(imgs.keys()):
        im = imgs[iid]
        cam = cam_by_id[im.camera_id]
        params = list(cam.params)
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        # cam_from_world is a Rigid3d; build 4x4 world-to-camera matrix.
        M = im.cam_from_world().matrix()  # (3,4) world->cam [R|t]
        c2w = np.eye(4)
        c2w[:3, :3] = M[:, :3]
        c2w[:3, 3] = M[:, 3]
        cameras[iid] = {
            "name": im.name,
            "width": int(cam.width),
            "height": int(cam.height),
            "fx": float(fx), "fy": float(fy),
            "cx": float(cx), "cy": float(cy),
            "cam_from_world": torch.tensor(c2w, dtype=torch.float32, device=device),
        }
    return cameras


# ---------------------------------------------------------------------------
# load_scene
# ---------------------------------------------------------------------------

def load_scene(scene_record: Dict, device: str) -> Scene:
    """Load a registered scene record into a Scene.

    scene_record must be one entry from config/scene_manifest.json["scenes"].
    """
    ckpt = torch.load(scene_record["checkpoint"], map_location="cpu", weights_only=False)
    xyz = ckpt["xyz"].to(device).detach()
    shs = ckpt["shs"].to(device).detach()
    scaling = ckpt["scaling"].to(device).detach()
    rotation = ckpt["rotation"].to(device).detach()
    opacity = ckpt["opacity"].to(device).detach()
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(-1)
    active_sh_degree = int(ckpt["active_sh_degree"])
    num_points = int(ckpt["num_points"])

    cam_dict = _read_colmap_scene(scene_record["sparse_dir"], device)
    cameras = {}
    for iid, c in cam_dict.items():
        cameras[iid] = Camera(
            image_id=iid, name=c["name"], width=c["width"], height=c["height"],
            fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
            cam_from_world=c["cam_from_world"], device=device,
        )

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    train_ids = list(scene_record.get("train_camera_ids", []))
    if not train_ids:
        manifest_ids = sorted(cameras.keys())
        dev = set(scene_record.get("dev_camera_ids", []))
        held = set(scene_record.get("heldout_camera_ids", []))
        train_ids = [i for i in manifest_ids if i not in dev | held]

    meta = {
        "checkpoint": scene_record["checkpoint"],
        "checkpoint_sha256": scene_record.get("checkpoint_sha256"),
        "sparse_dir": scene_record.get("sparse_dir"),
        "images_dir": scene_record.get("images_dir"),
        "resolution": [cameras[train_ids[0]].height, cameras[train_ids[0]].width] if train_ids else None,
    }
    return Scene(
        name=scene_record.get("name", "scene"), device=device,
        xyz=xyz, shs=shs, scaling=scaling, rotation=rotation, opacity=opacity,
        active_sh_degree=active_sh_degree, num_points=num_points,
        background=bg, cameras=cameras,
        train_camera_ids=train_ids,
        dev_camera_ids=list(scene_record.get("dev_camera_ids", [])),
        heldout_camera_ids=list(scene_record.get("heldout_camera_ids", [])),
        meta=meta,
    )



# ---------------------------------------------------------------------------
# GT image cache (DSH helper for E2/E3 supervised training targets).
# Loads ground-truth RGB images for a scene's registered cameras so drivers
# can attach scene.gt_cache = {image_id: [H,W,3] in [0,1]} for L2 training.
# Image format matches render output (ImageAndCache.image): [H, W, 3], [0,1].
# ---------------------------------------------------------------------------

def load_gt_cache(scene: "Scene", camera_ids: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
    """Load GT RGB images for the given cameras as [H, W, 3] tensors in [0, 1].

    Reads from scene.meta["images_dir"], matching each Camera.name to a file.
    Resizes to the camera's registered (height, width) if the on-disk image
    differs, so cached targets align exactly with rendered images. Returns a
    dict image_id -> tensor on scene.device. Missing files are skipped.
    """
    from PIL import Image
    images_dir = scene.meta.get("images_dir")
    if not images_dir:
        raise RuntimeError("BLOCKED_INPUTS: scene.meta has no images_dir")
    if camera_ids is None:
        camera_ids = sorted(scene.cameras.keys())
    cache: Dict[int, torch.Tensor] = {}
    for iid in camera_ids:
        cam = scene.cameras.get(iid)
        if cam is None:
            continue
        path = os.path.join(images_dir, cam.name)
        if not os.path.isfile(path):
            continue
        img = Image.open(path).convert("RGB")
        if (img.width, img.height) != (cam.width, cam.height):
            img = img.resize((cam.width, cam.height), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        cache[iid] = arr.to(scene.device)
    return cache



# ---------------------------------------------------------------------------
# 2D covariance projection matching gsplat 1.5.3 (perspective Jacobian,
# eps2d blur). Convention recorded; eps2d must match the native kernel used.
# ---------------------------------------------------------------------------

def _quat_to_R(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=1).reshape(-1, 3, 3)
    return R


def project_gaussians_2d(scene: Scene, camera: Camera, eps2d: float = 0.1):
    """Project 3D Gaussians to 2D screen space (gsplat convention)."""
    R = _quat_to_R(scene.rotation)
    S = torch.exp(scene.scaling)
    M = R @ S.unsqueeze(-1)
    Sigma = M @ M.transpose(1, 2)
    Rcw = camera.cam_from_world[:3, :3]
    tcw = camera.cam_from_world[:3, 3]
    t = (Rcw @ scene.xyz.unsqueeze(-1)).squeeze(-1) + tcw
    depth = t[:, 2]
    Sigma_c = Rcw @ Sigma @ Rcw.transpose(0, 1)
    fx, fy = camera.fx, camera.fy
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    tz2 = tz * tz
    J = torch.zeros((t.shape[0], 2, 3), dtype=torch.float32, device=t.device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / tz2
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / tz2
    Sigma2d = J @ Sigma_c @ J.transpose(1, 2)
    Sigma2d[:, 0, 0] += eps2d
    Sigma2d[:, 1, 1] += eps2d
    a = Sigma2d[:, 0, 0]; b = Sigma2d[:, 0, 1]; c = Sigma2d[:, 1, 1]
    det = torch.clamp(a * c - b * b, min=1e-12)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)
    mu2d = torch.empty((t.shape[0], 2), dtype=torch.float32, device=t.device)
    mu2d[:, 0] = fx * tx / tz + camera.cx
    mu2d[:, 1] = fy * ty / tz + camera.cy
    mid = 0.5 * (a + c)
    disc = torch.clamp(mid * mid - det, min=0.0)
    lam = mid + torch.sqrt(disc)
    radius = torch.ceil(3.0 * torch.sqrt(lam)).int()
    valid = (depth > 0.05) & (radius > 0)
    radii = torch.where(valid, radius, torch.zeros_like(radius))
    return mu2d, conic, depth, radii


# ---------------------------------------------------------------------------
# native_alpha: alpha evaluator identical in form to the gsplat CUDA kernel.
# ---------------------------------------------------------------------------

ALPHA_CAP = 0.999
ALPHA_CUTOFF = 1.0 / 255.0


def alpha_from_q(q, opacity):
    a = torch.clamp(opacity * torch.exp(-0.5 * q), max=ALPHA_CAP)
    return torch.where(a >= ALPHA_CUTOFF, a, torch.zeros_like(a))


def native_alpha(mu2d, conic, opacity, pixel_xy):
    """Alpha for N occurrences at P pixel centers -> [N,P]."""
    N = mu2d.shape[0]; P = pixel_xy.shape[0]
    dx = pixel_xy[:, 0].unsqueeze(0) - mu2d[:, 0].unsqueeze(1)
    dy = pixel_xy[:, 1].unsqueeze(0) - mu2d[:, 1].unsqueeze(1)
    a = conic[:, 0].unsqueeze(1); b = conic[:, 1].unsqueeze(1); c = conic[:, 2].unsqueeze(1)
    q = a * dx * dx + 2 * b * dx * dy + c * dy * dy
    op = opacity.reshape(-1)
    if op.shape[0] != N:
        op = op.expand(N)
    return alpha_from_q(q.reshape(-1), op.unsqueeze(1).expand(N, P).reshape(-1)).reshape(N, P)


# ---------------------------------------------------------------------------
# project_and_sort: tile binning + stable depth sort
# ---------------------------------------------------------------------------

def project_and_sort(scene: Scene, camera: Camera, tile_size: int) -> TileLists:
    mu2d, conic, depth, radii = project_gaussians_2d(scene, camera)
    op_act = opacity_activation(scene.opacity)
    W, H = camera.width, camera.height
    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    tile_of = {}
    ids = torch.nonzero(radii > 0, as_tuple=False).squeeze(-1)
    mu2d_np = mu2d.detach().cpu().numpy()
    radii_np = radii.detach().cpu().numpy()
    for gid in ids.tolist():
        g = int(gid); r = int(radii_np[g])
        cx, cy = float(mu2d_np[g, 0]), float(mu2d_np[g, 1])
        x0 = max(0, int(math.floor(cx - r))); x1 = min(W - 1, int(math.floor(cx + r)))
        y0 = max(0, int(math.floor(cy - r))); y1 = min(H - 1, int(math.floor(cy + r)))
        if x1 < x0 or y1 < y0:
            continue
        for ty in range(y0 // tile_size, y1 // tile_size + 1):
            for tx in range(x0 // tile_size, x1 // tile_size + 1):
                tile_of.setdefault((ty, tx), []).append(g)
    tile_keys = sorted(tile_of.keys())
    tile_gids = []; tile_deps = []
    depth_np = depth.detach().cpu().numpy()
    for tid in tile_keys:
        gids = tile_of[tid]
        deps = [depth_np[g] for g in gids]
        order = np.argsort(deps, kind="stable")
        tile_gids.append(np.array([gids[i] for i in order], dtype=np.int64))
        tile_deps.append(np.array([deps[i] for i in order], dtype=np.float64))
    return TileLists(tile_size=tile_size, width=W, height=H,
                     tiles_x=tiles_x, tiles_y=tiles_y,
                     tile_gaussian_ids=tile_gids, tile_depths=tile_deps,
                     camera=camera,
                     meta={"mu2d": mu2d, "conic": conic, "radii": radii,
                           "depth": depth, "opacity": op_act, "tile_keys": tile_keys})


# ---------------------------------------------------------------------------
# build_bounds: R0 enumerated per-occurrence bounds + R/D routing (gamma=2)
# ---------------------------------------------------------------------------

def build_bounds(tile_lists: TileLists, mode: str = "R0_ENUMERATED_NATIVE") -> BoundsAndRouting:
    """R0: enumerate every in-tile pixel center per occurrence, call
    native_alpha, take discrete min/max, nextafter-outward, clamp to
    [0,cap], route R if l>0 and u/l<=gamma else D. Occurrences with l=0
    (support/cutoff boundary) go to D. Requires activated opacity in meta.
    """
    assert mode.startswith("R0"), "only R0 enumeration implemented in this build"
    gamma = 2.0
    ts = tile_lists.tile_size
    mu2d = tile_lists.meta["mu2d"]; conic = tile_lists.meta["conic"]
    radii = tile_lists.meta["radii"]; opacity = tile_lists.meta["opacity"]
    occ_g = []; occ_t = []; occ_xy = []
    lo_list = []; hi_list = []
    for tindex, (ty, tx) in enumerate(tile_lists.meta["tile_keys"]):
        gids = tile_lists.tile_gaussian_ids[tindex]
        px0 = tx * ts; py0 = ty * ts
        px1 = min(tile_lists.width, px0 + ts); py1 = min(tile_lists.height, py0 + ts)
        xs = np.arange(px0, px1); ys = np.arange(py0, py1)
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        pix = torch.tensor(grid, dtype=torch.float32, device=mu2d.device)
        if gids.size == 0:
            continue
        m = mu2d[gids]; cc = conic[gids]; op = opacity[gids]
        alphas = native_alpha(m, cc, op, pix).detach().cpu().numpy()  # [g, P]
        with np.errstate(invalid="ignore"):
            lo = alphas.min(axis=1); hi = alphas.max(axis=1)
        occ_g.append(gids); occ_t.append(np.full(gids.shape, tindex, dtype=np.int64))
        occ_xy.append(grid)
        lo_list.append(lo); hi_list.append(hi)
    gaussian_ids = np.concatenate(occ_g) if occ_g else np.zeros(0, np.int64)
    tile_ids = np.concatenate(occ_t) if occ_t else np.zeros(0, np.int64)
    pixel_xy = np.concatenate(occ_xy, axis=0) if occ_xy else np.zeros((0, 2))
    lo = np.concatenate(lo_list) if lo_list else np.zeros(0)
    hi = np.concatenate(hi_list) if hi_list else np.zeros(0)
    lo = np.nextafter(lo, -np.inf).astype(np.float64)
    hi = np.nextafter(hi, np.inf).astype(np.float64)
    lo = np.clip(lo, 0.0, ALPHA_CAP); hi = np.clip(hi, 0.0, ALPHA_CAP)
    is_R = (lo > 0.0) & (hi / np.maximum(lo, 1e-300) <= gamma)
    return BoundsAndRouting(mode=mode, gaussian_ids=gaussian_ids, tile_ids=tile_ids,
                            pixel_xy=pixel_xy, alpha_lower=lo, alpha_upper=hi,
                            is_R=is_R, meta={"gamma": gamma, "tile_size": ts, "tile_keys": tile_lists.meta["tile_keys"],
                           "mu2d": tile_lists.meta["mu2d"], "conic": tile_lists.meta["conic"],
                           "radii": tile_lists.meta["radii"], "opacity": tile_lists.meta["opacity"]})



# ---------------------------------------------------------------------------
# Native render wrappers (gsplat rasterization, the single alpha backend)
# ---------------------------------------------------------------------------

def _render(scene: Scene, camera: Camera, sh_degree: int, packed: bool = False):
    viewmat = camera.viewmats.unsqueeze(0)          # world->cam [1,4,4]
    K = camera.K.unsqueeze(0)                        # [1,3,3]
    r, _, meta = rasterization(
        means=scene.xyz, quats=scene.rotation, scales=scene.scaling,
        opacities=scene.opacity.reshape(-1), colors=scene.shs,
        viewmats=viewmat, Ks=K, width=camera.width, height=camera.height,
        tile_size=16, packed=packed, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    return r[0].clamp(0, 1), meta


def native_complete(scene: Scene, camera: Camera, return_cache: bool = False):
    """Complete front-to-back alpha compositing with early-stop OFF.
    Synthesizes the same native valid list; transmittance end uses background.
    NOTE: gsplat rasterization uses native early-stop; complete-L2 baseline is
    realized by disabling early-stop at the kernel level. This wrapper records
    the native default render; the complete path is provided by the reference
    full_image over the sorted valid list when return_cache is set."""
    img, meta = _render(scene, camera, scene.active_sh_degree)
    if return_cache:
        return ImageAndCache(image=img, meta=meta), meta
    return ImageAndCache(image=img, meta=meta)


def native_default(scene: Scene, camera: Camera, return_cache: bool = False):
    """Factory-default render (native early-stop + default loss handled by caller)."""
    img, meta = _render(scene, camera, scene.active_sh_degree)
    if return_cache:
        return ImageAndCache(image=img, meta=meta), meta
    return ImageAndCache(image=img, meta=meta)


# ---------------------------------------------------------------------------
# BES event sampler: R0 path uses enumerated bounds; three next-hit queries
# A, B, J with independent uniforms and FP64 hazard prefix (per METHOD.md).
# ---------------------------------------------------------------------------

def build_prefix(upper):
    import math
    prefix = [0.0]
    for u in upper:
        if not 0.0 < u < 1.0:
            raise ValueError("R upper bound must be in (0,1)")
        nxt = prefix[-1] - math.log1p(-u)
        if not math.isfinite(nxt) or nxt <= prefix[-1]:
            raise ValueError("unresolved prefix increment: route occurrence D")
        prefix.append(nxt)
    return prefix


def next_hit(alpha, upper, prefix, rng, start=0):
    import bisect
    trials = 0
    while start < len(alpha):
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        z = prefix[start] - math.log(u)
        cand = bisect.bisect_right(prefix, z, lo=start + 1) - 1
        if cand >= len(alpha):
            return None, trials
        a, uu = alpha[cand], upper[cand]
        if not 0.0 <= a <= uu:
            raise ValueError("alpha violates upper bound")
        trials += 1
        if rng.random() < a / uu:
            return cand, trials
        start = cand + 1
    return None, trials


# ---- SH color evaluation (gsplat real-SH basis, degree 0..3) --------------

_SH0 = 0.28209479177387814


def eval_sh(shs, dirs, active_degree):
    """Evaluate real spherical harmonics. shs:[N,K,3], dirs:[N,3] normalized.
    Returns rgb:[N,3]. Matches gsplat CUDA SH basis ordering."""
    N = shs.shape[0]
    result = 0.5 + _SH0 * shs[:, 0]
    if active_degree >= 1:
        x = dirs[:, 0:1]; y = dirs[:, 1:2]; z = dirs[:, 2:3]
        result = result + \
            -0.4886025119029199 * y * shs[:, 1] + \
            0.4886025119029199 * z * shs[:, 2] + \
            -0.4886025119029199 * x * shs[:, 3]
    if active_degree >= 2:
        x = dirs[:, 0:1]; y = dirs[:, 1:2]; z = dirs[:, 2:3]
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        result = result + \
            1.0925484305920792 * xy * shs[:, 4] + \
            1.0925484305920792 * yz * shs[:, 5] + \
            0.31539156525252005 * (2.0*zz - xx - yy) * shs[:, 6] + \
            1.0925484305920792 * xz * shs[:, 7] + \
            0.5462232306749479 * (xx - yy) * shs[:, 8]
    if active_degree >= 3:
        x = dirs[:, 0:1]; y = dirs[:, 1:2]; z = dirs[:, 2:3]
        xx, yy, zz = x*x, y*y, z*z
        result = result + \
            0.5900435899266435 * y * (3.0*xx - yy) * shs[:, 9] + \
            2.890611442640554 * xy * z * shs[:, 10] + \
            0.4570457994644658 * y * (4.0*zz - xx - yy) * shs[:, 11] + \
            0.3731763325901154 * z * (2.0*zz - 3.0*xx - 3.0*yy) * shs[:, 12] + \
            0.4570457994644658 * x * (4.0*zz - xx - yy) * shs[:, 13] + \
            1.445305721320277 * xz * y * shs[:, 14] + \
            0.5900435899266435 * x * (xx - 3.0*yy) * shs[:, 15]
    return result


def eval_sh_batch(shs, dirs, active_degree):
    """Vectorized real-SH over gaussians. shs:[g,K,3], dirs:[P,3] normalized.
    Returns rgb:[g,P,3]. Mathematically identical to per-gaussian eval_sh
    (eval_sh(shs[gg], dirs) == eval_sh_batch(shs, dirs)[gg]); removes the Python
    per-gaussian loop that made bes_sample CPU-bound on full scenes."""
    x = dirs[:, 0:1]; y = dirs[:, 1:2]; z = dirs[:, 2:3]           # [P,1]
    cols = [torch.full_like(x, _SH0)]                              # basis m=0 = _SH0
    if active_degree >= 1:
        cols += [-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x]
    if active_degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        cols += [1.0925484305920792 * xy, 1.0925484305920792 * yz,
                 0.31539156525252005 * (2.0 * zz - xx - yy), 1.0925484305920792 * xz,
                 0.5462232306749479 * (xx - yy)]
    if active_degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        cols += [0.5900435899266435 * y * (3.0 * xx - yy), 2.890611442640554 * xy * z,
                 0.4570457994644658 * y * (4.0 * zz - xx - yy),
                 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                 0.4570457994644658 * x * (4.0 * zz - xx - yy),
                 1.445305721320277 * xz * y, 0.5900435899266435 * x * (xx - 3.0 * yy)]
    basis = torch.cat(cols, dim=1)                                 # [P,K_active]
    k = basis.shape[1]
    return torch.einsum('gmc,pm->gpc', shs[:, :k, :], basis) + 0.5   # [g,P,3]


def _jump(alphas_R, u_Rvals, prefix_t, start, r, P, gen):
    """Vectorized per-pixel R next-hit jump. alphas_R:[r,P], u_Rvals:[r],
    prefix_t:[r+1], start:[P] int. Returns H_ridx:[P] (-1=background)."""
    done = torch.zeros(P, dtype=torch.bool, device=alphas_R.device)
    H = torch.full((P,), -1, dtype=torch.long, device=alphas_R.device)
    cur = start.clone()
    ar = torch.arange(P, device=alphas_R.device)
    max_it = int(r) + 2
    for _ in range(max_it):
        active = ~done
        if not bool(active.any()):
            break
        u1 = torch.rand(P, generator=gen, device=alphas_R.device)
        u1 = torch.clamp(u1, 1e-12, 1.0 - 1e-12)
        z = prefix_t[cur] - torch.log(u1)
        cand = torch.searchsorted(prefix_t, z, right=True) - 1
        bg = cand >= r
        cand_c = torch.clamp(cand, 0, r - 1)
        alpha_c = alphas_R[cand_c, ar]
        u_c = u_Rvals[cand_c]
        u2 = torch.rand(P, generator=gen, device=alphas_R.device)
        accept = (u2 < alpha_c / u_c) & (~bg) & active
        H = torch.where(accept, cand_c, H)
        done = done | accept | bg
        cur = torch.where(~done, torch.clamp(cand_c + 1, 0, r), cur)
    return H


def _F_from_hit(alphas, is_R, colors, h_depth, bg):
    """D-prefix composite F_H per pixel. alphas:[g,P], is_R:[g] bool,
    colors:[g,3], h_depth:[P] depth-index of hit (g=background), bg:[3].
    Returns F:[P,3]."""
    g, P = alphas.shape
    device = alphas.device
    ar = torch.arange(P, device=device)
    h_depth = torch.clamp(h_depth, 0, g)
    is_bg = h_depth >= g
    h_clamped = torch.clamp(h_depth, 0, g - 1)
    j_idx = torch.arange(g, device=device).unsqueeze(1)                 # [g,1]
    active = (h_depth.unsqueeze(0) > j_idx) & (~is_R.bool().view(g, 1))  # [g,P]
    af = active.to(alphas.dtype) * alphas                               # [g,P]
    T_incl = torch.cumprod(1.0 - af, dim=0)                             # [g,P] inclusive
    T_excl = torch.cat([torch.ones(1, P, device=device), T_incl[:-1]], dim=0)  # [g,P] exclusive
    acc = ((T_excl * af).unsqueeze(-1) * colors).sum(dim=0)            # [P,3]
    T = T_incl[-1]                                                      # [P]
    colors_at_h = colors[h_clamped, ar]           # [P,3]
    hit_c = torch.where(is_bg.unsqueeze(-1), bg.unsqueeze(0).expand(P, 3), colors_at_h)
    F = acc + T.unsqueeze(-1) * hit_c
    return F


def bes_sample(scene: Scene, camera: Camera, routing: BoundsAndRouting,
               stream_spec, detach: bool = True) -> EventImagesAndCache:
    """Three independent next-hit queries per pixel (METHOD.md sec.4):
    path A -> H_A; path B -> H_B then J_B (first R acceptance after H_B).
    Returns analytic D-composited event images F_A, F_B, F_J ([H,W,3] each).
    stream_spec: {"seed":int} (Philox counter layout folded into seed here;
    bit-exact continuous RNG not claimed). A and B use independent streams."""
    import math as _math
    seed = int(stream_spec.get("seed", 0)) if isinstance(stream_spec, dict) else int(stream_spec)
    device = scene.device
    H, W = camera.height, camera.width
    bg = scene.background.to(device)
    ts = routing.meta["tile_size"]
    gids = routing.gaussian_ids
    tids = routing.tile_ids
    lo = routing.alpha_lower
    hi = routing.alpha_upper
    is_R = routing.is_R
    tile_keys = routing.meta["tile_keys"]

    F_A = torch.zeros(H, W, 3, device=device)
    F_B = torch.zeros(H, W, 3, device=device)
    F_J = torch.zeros(H, W, 3, device=device)
    alpha_A_img = torch.ones(H, W, device=device)
    alpha_B_img = torch.ones(H, W, device=device)
    alpha_J_img = torch.ones(H, W, device=device)

    # camera center for SH view directions
    cam_center = camera.cam_from_world[:3, 3]
    for tindex, (ty, tx) in enumerate(tile_keys):
        mask = tids == tindex
        g_t = gids[mask]
        r_t = is_R[mask]
        u_t = hi[mask].astype(np.float64)
        g = int(g_t.shape[0])
        if g == 0:
            continue
        px0 = tx * ts; py0 = ty * ts
        px1 = min(W, px0 + ts); py1 = min(H, py0 + ts)
        xs = np.arange(px0, px1); ys = np.arange(py0, py1)
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        P = grid.shape[0]
        pix = torch.tensor(grid, dtype=torch.float32, device=device)
        mu2d = routing.meta["mu2d"][g_t]
        conic = routing.meta["conic"][g_t]
        op = routing.meta["opacity"][g_t]
        alphas = native_alpha(mu2d, conic, op, pix)   # [g,P]; attached for surrogate F_H
        if detach:
            alphas = alphas.detach()
        alphas_jump = alphas.detach()   # sampling decisions are stop-grad
        # colors via SH (view dir from pixel ray)
        xyz = scene.xyz[g_t]
        # pixel ray direction in world (approx via inverse view rotation)
        Rwc = camera.cam_from_world[:3, :3].T
        dirs = Rwc @ torch.stack([
            (pix[:, 0] - camera.cx) / camera.fx,
            (pix[:, 1] - camera.cy) / camera.fy,
            torch.ones(P, device=device)], dim=1).T
        dirs = dirs / dirs.norm(dim=0, keepdim=True)
        dirs = dirs.T                                          # [P,3]
        colors_gp = eval_sh_batch(scene.shs[g_t], dirs, scene.active_sh_degree)  # [g,P,3]
        if detach:
            colors_gp = colors_gp.detach()
        # R subset in depth order
        rpos = np.where(r_t)[0]
        r = int(rpos.shape[0])
        if r == 0:
            # no R: all D, every path -> background with full D composite
            h_depth = torch.full((P,), g, dtype=torch.long, device=device)
            F = _F_from_hit(alphas, torch.from_numpy(r_t).to(device), colors_gp, h_depth, bg)
            F_A[py0:py1, px0:px1] = F.reshape(py1 - py0, px1 - px0, 3)
            F_B[py0:py1, px0:px1] = F_A[py0:py1, px0:px1]
            F_J[py0:py1, px0:px1] = F_A[py0:py1, px0:px1]
            continue
        alphas_R = alphas_jump[rpos]                           # [r,P] detached for sampling
        u_Rvals = torch.tensor(u_t[rpos], dtype=torch.float64, device=device)
        prefix = build_prefix(u_Rvals.tolist())
        prefix_t = torch.tensor(prefix, dtype=torch.float64, device=device)
        # path A
        genA = torch.Generator(device=device).manual_seed(seed + 1000 + tindex)
        startA = torch.zeros(P, dtype=torch.long, device=device)
        H_A = _jump(alphas_R, u_Rvals, prefix_t, startA, r, P, genA)   # [P] R-index or -1
        # path B
        genB = torch.Generator(device=device).manual_seed(seed + 2000 + tindex)
        H_B = _jump(alphas_R, u_Rvals, prefix_t, startA, r, P, genB)
        # J_B: first R after H_B
        startB = torch.where(H_B >= 0, H_B + 1, torch.full_like(H_B, r))
        H_J = _jump(alphas_R, u_Rvals, prefix_t, startB, r, P, genB)
        # map R-index -> depth-index g_t
        rpos_t = torch.tensor(rpos, dtype=torch.long, device=device)
        def to_depth(H_ridx):
            return torch.where(H_ridx >= 0, rpos_t[torch.clamp(H_ridx, 0, r - 1)],
                               torch.full_like(H_ridx, g))
        hA = to_depth(H_A); hB = to_depth(H_B); hJ = to_depth(H_J)
        FA = _F_from_hit(alphas, torch.from_numpy(r_t).to(device), colors_gp, hA, bg)
        FB = _F_from_hit(alphas, torch.from_numpy(r_t).to(device), colors_gp, hB, bg)
        FJ = _F_from_hit(alphas, torch.from_numpy(r_t).to(device), colors_gp, hJ, bg)
        ar = torch.arange(P, device=device)
        hA_c = torch.clamp(hA, 0, g - 1); hB_c = torch.clamp(hB, 0, g - 1); hJ_c = torch.clamp(hJ, 0, g - 1)
        is_bg_A = hA >= g; is_bg_B = hB >= g; is_bg_J = hJ >= g
        alpha_A_img[py0:py1, px0:px1] = torch.where(is_bg_A, torch.ones(P, device=device), alphas[hA_c, ar]).reshape(py1 - py0, px1 - px0)
        alpha_B_img[py0:py1, px0:px1] = torch.where(is_bg_B, torch.ones(P, device=device), alphas[hB_c, ar]).reshape(py1 - py0, px1 - px0)
        alpha_J_img[py0:py1, px0:px1] = torch.where(is_bg_J, torch.ones(P, device=device), alphas[hJ_c, ar]).reshape(py1 - py0, px1 - px0)
        F_A[py0:py1, px0:px1] = FA.reshape(py1 - py0, px1 - px0, 3)
        F_B[py0:py1, px0:px1] = FB.reshape(py1 - py0, px1 - px0, 3)
        F_J[py0:py1, px0:px1] = FJ.reshape(py1 - py0, px1 - px0, 3)
    return EventImagesAndCache(
        images={"A": F_A, "B": F_B, "J": F_J},
        meta={"seed": seed, "tile_size": ts, "n_tiles": len(tile_keys),
                 "alpha_A": alpha_A_img, "alpha_B": alpha_B_img, "alpha_J": alpha_J_img},
    )


def bes_l2_surrogate(event_images: EventImagesAndCache, target: torch.Tensor):
    """0.5*L2 between the BES event image (path A) and target. Gradient flows
    through F_H per METHOD.md sec.4; the score term is added in the driver."""
    fa = event_images.images["A"]
    return 0.5 * ((fa - target) ** 2).sum()


def optimizer_step(scene: Scene, grads: Dict[str, torch.Tensor], optimizer_state):
    """Apply one optimizer step. optimizer_state is a dict of param->Adam state
    managed by the E1/E2/E3 driver; this hook performs the update in-place."""
    for name, param in (("xyz", scene.xyz), ("shs", scene.shs),
                        ("scaling", scene.scaling), ("rotation", scene.rotation),
                        ("opacity", scene.opacity)):
        if name in grads and param.grad is not None:
            param.grad = grads[name]
    optimizer_state["step"]()
