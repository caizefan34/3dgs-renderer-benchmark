#!/usr/bin/env python3
"""
C18 — Quantitative Decomposition Gate.

Measures consecutive-iteration Gaussian→Tile membership stability to
determine whether incremental intersection rebuild is computationally viable.

No renderer modifications. No CUDA implementation. No literature survey.

Self-contained — does not import from scripts.epic05.phase7 to avoid
Python path issues on remote execution.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Path setup
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

from gsplat import rasterization, __version__ as gsplat_version
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras


def _load_phase7_module(name: str):
    """Load the canonical Phase7 module by path, avoiding `scripts` import shadowing."""
    path = ROOT / "scripts" / "epic05" / "phase7" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"c18_phase7_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical Phase7 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_canonical_dataset = _load_phase7_module("dataset")
_canonical_model = _load_phase7_module("gaussian_model")
_canonical_loss = _load_phase7_module("loss")
CanonicalGTDataset = _canonical_dataset.GTDataset
CanonicalGaussianModel = _canonical_model.GaussianModel
canonical_combined_loss = _canonical_loss.combined_loss

torch.backends.cudnn.deterministic = True


# ── Inlined GTDataset (avoids import from scripts.epic05.phase7) ──────────

class GTDataset:
    """Dataset loading real GT images with matched cameras. (Inlined from dataset.py)"""

    def __init__(self, scene: str, repo_root: str | Path, resolution: str = "1080p",
                 device: str = "cuda"):
        self.scene = scene
        self.repo_root = Path(repo_root).resolve()
        self.device = device

        RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
        target_res = RESOLUTIONS.get(resolution, (1920, 1080))
        self.target_width, self.target_height = target_res

        self.official_dir = self.repo_root / "data" / "official" / "mipnerf360" / scene
        self.gt_dir = self.repo_root / "data" / "datasets" / "mipnerf360" / scene / "images"

        camera_path = self.official_dir / "cameras.json"
        self._cameras = load_cameras_from_json(str(camera_path), device="cpu")
        self._cameras = resize_cameras(self._cameras, self.target_width, self.target_height)

        self._image_index = {}
        for fpath in self.gt_dir.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in (".jpg", ".jpeg", ".png"):
                self._image_index[fpath.stem] = fpath

        self._valid_cameras = []
        self._valid_image_paths = []
        for cam in self._cameras:
            img_name = getattr(cam, "image_name", None) or getattr(cam, "img_name", None)
            if img_name and Path(img_name).stem in self._image_index:
                self._valid_cameras.append(cam)
                self._valid_image_paths.append(self._image_index[Path(img_name).stem])

        print(f"  [DS] {len(self._valid_cameras)} cameras matched")
        from PIL import Image
        self._gt_cache = [None] * len(self._valid_cameras)
        for i, path in enumerate(self._valid_image_paths):
            with Image.open(path) as source:
                if source.width != self.target_width or source.height != self.target_height:
                    source = source.resize((self.target_width, self.target_height), Image.LANCZOS)
                rgba_np = np.array(source.convert("RGBA"), dtype=np.uint8)
            self._gt_cache[i] = torch.from_numpy(rgba_np)
        total_mb = sum(t.numel() for t in self._gt_cache) / (1024**2)
        print(f"  [DS] All {len(self._valid_cameras)} images cached ({total_mb:.0f} MB)")

    def __len__(self):
        return len(self._valid_cameras)

    def get_camera(self, index: int):
        cam = self._valid_cameras[index]
        for attr in ["viewmatrix", "projmatrix", "camera_center",
                      "world_view_transform", "full_proj_transform", "K"]:
            t = getattr(cam, attr)
            if isinstance(t, torch.Tensor):
                setattr(cam, attr, t.to(self.device))
        return cam

    def get_gt_image(self, index: int) -> torch.Tensor:
        rgba = self._gt_cache[index]
        return rgba.to(self.device, non_blocking=True, dtype=torch.float32)[..., :3].div_(255.0).contiguous()

    def get_item(self, index: int):
        return self.get_camera(index), self.get_gt_image(index)


# ── Inlined GaussianModel ────────────────────────────────────────────────

class GaussianModel(torch.nn.Module):
    """(Simplified) Gaussian parameter holder with densification/pruning.
    Inlined from phase7/gaussian_model.py — same API, no extra dependencies."""

    def __init__(self, num_points: int, sh_degree: int = 0, max_sh_degree: int = 3,
                 device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.sh_degree = sh_degree
        self.max_sh_degree = max_sh_degree
        self.num_sh_coeffs = (max_sh_degree + 1) ** 2
        self.num_points = num_points
        self._xyz_grad_accum = None
        self._denf_steps = 0

    def init_from_sfm(self, xyz, opacity_logit=None, scales_log=None,
                      rotations_raw=None, shs=None):
        N = xyz.shape[0]
        self.xyz = torch.nn.Parameter(xyz.to(self.device).contiguous())

        if opacity_logit is not None:
            self.opacity = torch.nn.Parameter(opacity_logit.to(self.device).contiguous())
        else:
            self.opacity = torch.nn.Parameter(
                torch.logit(torch.full((N, 1), 0.1, device=self.device))
            )

        if scales_log is not None:
            self.scales = torch.nn.Parameter(scales_log.to(self.device).contiguous())
        else:
            self.scales = torch.nn.Parameter(
                torch.log(torch.full((N, 3), 0.01, device=self.device))
            )

        if rotations_raw is not None:
            self.rotations = torch.nn.Parameter(rotations_raw.to(self.device).contiguous())
        else:
            self.rotations = torch.nn.Parameter(
                torch.zeros(N, 4, device=self.device)
            )
            self.rotations.data[:, 0] = 1.0

        if shs is not None:
            self.shs = torch.nn.Parameter(shs.to(self.device).contiguous())
            sd = round(math.sqrt(shs.shape[1]) - 1)
            self.sh_degree = int(sd)
        else:
            self.shs = torch.nn.Parameter(
                torch.zeros(N, self.num_sh_coeffs, 3, device=self.device)
            )
        self.num_points = N

    def forward(self) -> dict:
        return {
            "xyz": self.xyz,
            "rotations": torch.nn.functional.normalize(self.rotations, dim=-1).contiguous(),
            "scales": torch.exp(self.scales).contiguous(),
            "opacity": torch.sigmoid(self.opacity).squeeze(-1).contiguous(),
            "shs": self.shs.contiguous(),
            "num_points": self.xyz.shape[0],
            "sh_degree": self.sh_degree,
        }

    def accumulate_positional_gradient(self):
        if self.xyz.grad is None:
            return
        grad = self.xyz.grad.detach()
        grad_norm = grad.norm(dim=-1)
        if self._xyz_grad_accum is None or self._xyz_grad_accum.shape != grad_norm.shape:
            self._xyz_grad_accum = grad_norm
            self._denf_steps = 1
        else:
            self._xyz_grad_accum = self._xyz_grad_accum + grad_norm
            self._denf_steps += 1

    @torch.no_grad()
    def densification(self, grad_threshold: float = 2e-4, clone_max_screen_size: float = 100.0,
                      split_max_screen_size: float = 100.0) -> dict:
        if self._xyz_grad_accum is None or self._denf_steps == 0:
            return {"cloned": 0, "split": 0, "removed": 0}
        avg_grad = self._xyz_grad_accum / self._denf_steps
        high_grad_mask = avg_grad >= grad_threshold
        self._xyz_grad_accum = None
        self._denf_steps = 0

        scales_activated = torch.exp(self.scales).detach()
        median_scale = scales_activated.median(dim=0).values
        is_small = (scales_activated <= median_scale).all(dim=-1)
        clone_mask = high_grad_mask & is_small
        split_mask = high_grad_mask & (~is_small)
        clone_count, split_count = clone_mask.sum().item(), split_mask.sum().item()
        if clone_count == 0 and split_count == 0:
            return {"cloned": 0, "split": 0, "removed": 0}

        orig_xyz = self.xyz.detach()
        orig_rot = self.rotations.detach()
        orig_scales = self.scales.detach()
        orig_opacity = self.opacity.detach()
        orig_shs = self.shs.detach()

        new_parts = {"xyz": [], "rotations": [], "scales": [], "opacity": [], "shs": []}
        if clone_count > 0:
            noise = torch.randn_like(orig_xyz[clone_mask]) * 0.01 * torch.exp(orig_scales[clone_mask])
            new_parts["xyz"].append(orig_xyz[clone_mask] + noise)
            new_parts["rotations"].append(orig_rot[clone_mask])
            new_parts["scales"].append(orig_scales[clone_mask])
            new_parts["opacity"].append(orig_opacity[clone_mask])
            new_parts["shs"].append(orig_shs[clone_mask])
        if split_count > 0:
            new_parts["xyz"].append(orig_xyz[split_mask])
            new_parts["rotations"].append(orig_rot[split_mask])
            new_parts["scales"].append(orig_scales[split_mask] - math.log(2.0))
            new_parts["opacity"].append(orig_opacity[split_mask])
            new_parts["shs"].append(orig_shs[split_mask])
            # Original ones get halved too — split the original
            self.xyz.data[split_mask] = orig_xyz[split_mask]
            self.rotations.data[split_mask] = orig_rot[split_mask]
            self.scales.data[split_mask] = orig_scales[split_mask] - math.log(2.0)
            self.opacity.data[split_mask] = orig_opacity[split_mask]
            self.shs.data[split_mask] = orig_shs[split_mask]

        if new_parts["xyz"]:
            self.xyz = torch.nn.Parameter(torch.cat([self.xyz] + new_parts["xyz"], dim=0).contiguous())
            self.rotations = torch.nn.Parameter(torch.cat([self.rotations] + new_parts["rotations"], dim=0).contiguous())
            self.scales = torch.nn.Parameter(torch.cat([self.scales] + new_parts["scales"], dim=0).contiguous())
            self.opacity = torch.nn.Parameter(torch.cat([self.opacity] + new_parts["opacity"], dim=0).contiguous())
            self.shs = torch.nn.Parameter(torch.cat([self.shs] + new_parts["shs"], dim=0).contiguous())
            self.num_points = self.xyz.shape[0]
        return {"cloned": clone_count, "split": split_count, "removed": 0}

    @torch.no_grad()
    def prune(self, opacity_threshold: float = 0.005) -> int:
        opacities = torch.sigmoid(self.opacity).detach().squeeze(-1)
        prune_mask = opacities < opacity_threshold
        removed = prune_mask.sum().item()
        if removed == 0:
            return 0
        keep_mask = ~prune_mask
        self.xyz = torch.nn.Parameter(self.xyz[keep_mask].contiguous())
        self.rotations = torch.nn.Parameter(self.rotations[keep_mask].contiguous())
        self.scales = torch.nn.Parameter(self.scales[keep_mask].contiguous())
        self.opacity = torch.nn.Parameter(self.opacity[keep_mask].contiguous())
        self.shs = torch.nn.Parameter(self.shs[keep_mask].contiguous())
        self.num_points = self.xyz.shape[0]
        self._xyz_grad_accum = None
        self._denf_steps = 0
        return removed

    @torch.no_grad()
    def prune_and_reset(self, opacity_threshold: float = 0.005,
                        reset_interval: int = 3000, current_step: int = 0) -> int:
        removed = self.prune(opacity_threshold)
        if current_step > 0 and current_step % reset_interval == 0:
            opacities = torch.sigmoid(self.opacity).detach().squeeze(-1)
            near_threshold = (opacities < opacity_threshold * 10) & (opacities >= opacity_threshold)
            reset_count = near_threshold.sum().item()
            if reset_count > 0:
                reset_val = torch.logit(torch.full((reset_count, 1), opacity_threshold * 2, device=self.device))
                self.opacity.data[near_threshold] = reset_val
        return removed


# ── Loss function ────────────────────────────────────────────────────────

def combined_loss(img, gt_img, lambda_dssim=0.2):
    """L1 + D-SSIM loss. Inlined from phase7/loss.py."""
    l1 = torch.abs(img - gt_img).mean()
    # Simplified SSIM
    C1, C2 = 0.01**2, 0.03**2
    mu1 = img.mean(dim=(0, 1), keepdim=True)
    mu2 = gt_img.mean(dim=(0, 1), keepdim=True)
    sig1 = ((img - mu1) ** 2).mean(dim=(0, 1), keepdim=True)
    sig2 = ((gt_img - mu2) ** 2).mean(dim=(0, 1), keepdim=True)
    sig12 = ((img - mu1) * (gt_img - mu2)).mean(dim=(0, 1), keepdim=True)
    ssim = (2 * mu1 * mu2 + C1) * (2 * sig12 + C2) / (
        mu1 ** 2 + mu2 ** 2 + C1
    ) / (sig1 + sig2 + C2)
    dssim = (1 - ssim.mean()) * 0.5
    return {"loss": l1 * (1 - lambda_dssim) + dssim * lambda_dssim, "l1": l1, "dssim": dssim}


# ── Data extraction ──────────────────────────────────────────────────────

def extract_visible_data(
    info: dict,
    xyz: torch.Tensor | None = None,
    capture_state: bool = False,
) -> dict:
    """Extract exact membership data and sparse projection state samples."""
    gids = info["gaussian_ids"].long()
    flat = info["flatten_ids"].long()
    offsets = info["isect_offsets"][0].reshape(-1).long()
    n = flat.numel()
    nt = offsets.numel()

    ends = torch.cat((offsets[1:], offsets.new_tensor([n])))
    counts = ends - offsets
    tile_ids = torch.repeat_interleave(
        torch.arange(nt, dtype=torch.long, device=offsets.device), counts
    )

    entry_gids = gids[flat]
    MAX_TILES = 1 << 24
    encoded = torch.sort(entry_gids * MAX_TILES + tile_ids).values
    visible_gids, _ = torch.sort(gids)

    return {
        "entry_gids": entry_gids,
        "tile_ids": tile_ids,
        "encoded": encoded,
        "visible_gids": visible_gids,
        "n_entries": int(n),
        "n_visible": int(gids.numel()),
        "n_tiles_active": int((counts > 0).sum().item()),
        # These arrays are retained only for the sparse per-Gaussian samples.
        "sample_state": ({
            "gids": gids.detach().cpu().numpy(),
            "means2d": info["means2d"].detach().cpu().numpy(),
            "radii": info["radii"].detach().cpu().numpy(),
            "conics": info["conics"].detach().cpu().numpy(),
            "xyz": xyz[gids].detach().cpu().numpy() if xyz is not None else None,
        } if capture_state else None),
    }


def compute_intersection_overlap(d0: dict, d1: dict) -> dict:
    """Exact GPU intersection of sorted encoded (gaussian_id, tile_id) pairs.

    Do not loop over GPU scalar tensors here: that forces one CUDA synchronization
    per intersection record and made the initial C18 probe appear hung.
    """
    enc0, enc1 = d0["encoded"], d1["encoded"]
    n0, n1 = enc0.numel(), enc1.numel()
    if n0 == 0 or n1 == 0:
        common = 0
    else:
        pos = torch.searchsorted(enc1, enc0)
        in_range = pos < n1
        equal = torch.zeros_like(in_range)
        equal[in_range] = enc1[pos[in_range]] == enc0[in_range]
        common = int(equal.sum().item())

    n_union = n0 + n1 - common
    return {
        "intersection_overlap": float(common / n_union) if n_union > 0 else 1.0,
        "reusable_intersection_ratio": float(common / n1) if n1 > 0 else 1.0,
        "new_intersection_ratio": float((n1 - common) / n1) if n1 > 0 else 0.0,
        "removed_intersection_ratio": float((n0 - common) / n0) if n0 > 0 else 0.0,
        "n_common_entries": int(common),
    }


def compute_membership_stability_per_gaussian(d0: dict, d1: dict, tile_size: int = 16) -> dict:
    """Exact per-Gaussian membership and sampled state/cause decomposition.

    Valid only when no densification/pruning event occurred between snapshots."""
    gids0, tiles0 = d0["entry_gids"], d0["tile_ids"]
    gids1, tiles1 = d1["entry_gids"], d1["tile_ids"]

    MAX_TILES = 1 << 24
    packed0, order0 = torch.sort(gids0 * MAX_TILES + tiles0)
    order1 = torch.argsort(gids1 * MAX_TILES + tiles1)
    sgid0 = gids0[order0]
    stile0 = tiles0[order0]
    sgid1 = gids1[order1]
    stile1 = tiles1[order1]

    sgid0_np = sgid0.cpu().numpy()
    stile0_np = stile0.cpu().numpy()
    sgid1_np = sgid1.cpu().numpy()
    stile1_np = stile1.cpu().numpy()

    def build_tile_dict(sgid, stile):
        d = {}
        g_start = 0
        n = len(sgid)
        for i in range(n + 1):
            if i == n or sgid[i] != sgid[g_start]:
                d[int(sgid[g_start])] = set(stile[g_start:i].tolist())
                g_start = i
        return d

    tset0 = build_tile_dict(sgid0_np, stile0_np)
    tset1 = build_tile_dict(sgid1_np, stile1_np)

    common_gids = set(tset0.keys()) & set(tset1.keys())
    unchanged = sum(1 for g in common_gids if tset0[g] == tset1[g])
    changed = len(common_gids) - unchanged
    total = len(common_gids)

    # Visibility and state/cause statistics are derived from gsplat's projected
    # per-visible-Gaussian arrays, not inferred from membership pair counts.
    v0, v1 = set(tset0), set(tset1)
    visibility_changed = len(v0 ^ v1)
    visibility_changed_ratio = visibility_changed / max(len(v0 | v1), 1)
    state0, state1 = d0.get("sample_state"), d1.get("sample_state")
    causes = {k: 0 for k in ("center_crosses_tile_boundary", "radius_change", "covariance_footprint_change", "numerical_or_other")}
    pos_delta = []; center_delta = []; radius_delta = []
    if state0 is not None and state1 is not None:
        m0 = {int(g): i for i, g in enumerate(state0["gids"])}
        m1 = {int(g): i for i, g in enumerate(state1["gids"])}
        for g in common_gids:
            i0, i1 = m0[g], m1[g]
            c0, c1 = state0["means2d"][i0], state1["means2d"][i1]
            r0, r1 = state0["radii"][i0], state1["radii"][i1]
            pos_delta.append(float(np.linalg.norm(state1["xyz"][i1] - state0["xyz"][i0])))
            center_delta.append(float(np.linalg.norm(c1 - c0)))
            radius_delta.append(float(np.linalg.norm(r1.astype(np.float64) - r0.astype(np.float64))))
            if g in common_gids and tset0[g] != tset1[g]:
                if np.any(np.floor(c0 / tile_size) != np.floor(c1 / tile_size)):
                    causes["center_crosses_tile_boundary"] += 1
                elif np.any(r0 != r1):
                    causes["radius_change"] += 1
                elif not np.allclose(state0["conics"][i0], state1["conics"][i1], rtol=0.0, atol=1e-7):
                    causes["covariance_footprint_change"] += 1
                else:
                    causes["numerical_or_other"] += 1
    return {
        "n_common_gaussians": total,
        "stable_membership_ratio": float(unchanged / total) if total > 0 else 0.0,
        "changed_membership_ratio": float(changed / total) if total > 0 else 0.0,
        "visibility_changed_ratio": float(visibility_changed_ratio),
        "position_change": stats(pos_delta),
        "projected_center_change_px": stats(center_delta),
        "projected_radius_change_px": stats(radius_delta),
        "membership_change_causes": causes,
    }


def stats(arr):
    if not arr:
        return None
    a = np.array(arr)
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()), "max": float(a.max()),
        "n": int(len(a)),
    }


def run_decomposition(args):
    device = torch.device("cuda")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        print(f"[C18] WARNING: GPU={torch.cuda.get_device_name(0)}, NOT A100")
    else:
        print(f"[C18] GPU: A100-PCIE-40GB confirmed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    out = ROOT / "results" / "phase-c18"
    out.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports" / "phase-c18"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    print(f"[C18] Loading scene '{args.scene}' at {args.resolution}...")
    ds = CanonicalGTDataset(args.scene, str(ROOT), resolution=args.resolution, device=device)
    sfm = load_ply(
        str(ROOT / "data" / "official" / "mipnerf360" / args.scene / "point_cloud.ply"),
        device=device,
    )
    print(f"[C18] SfM points: {sfm['xyz'].shape[0]:,}")

    # ── Init model ──
    model = CanonicalGaussianModel(sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        sfm["xyz"],
        torch.logit(torch.full((sfm["xyz"].shape[0], 1), 0.1, device=device)),
        sfm["scales"], sfm["rotations"], sfm["shs"],
    )
    extent = float(sfm["xyz"].norm(dim=-1).max().item())
    print(f"[C18] Extent: {extent:.2f}")

    def make_optim():
        return torch.optim.Adam([
            {"params": [model.xyz], "lr": 1.6e-4 * extent, "eps": 1e-15},
            {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15},
            {"params": [model.scales], "lr": 5e-3, "eps": 1e-15},
            {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15},
            {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15},
        ])

    optim = make_optim()

    # Fixed evaluation camera (camera 0)
    eval_cam, eval_gt = ds.get_item(0)
    print(f"[C18] Fixed camera 0: {eval_cam.image_width}x{eval_cam.image_height}")
    print(f"[C18] Running {args.steps} steps, tile_size={args.tile_size}...")

    # ── Data collection ──
    step_records = {}
    topology_events = []
    n_gaussians_history = []

    pair_metrics_no_topo = []
    pair_metrics_topo = []
    pair_metrics_lag2 = []
    pair_metrics_lag4 = []

    t0 = time.perf_counter()

    for step in range(args.steps):
        # Training step with alternating camera
        train_cam_idx = step % len(ds)
        train_cam, train_target = ds.get_item(train_cam_idx)

        d = model.forward()
        train_img, _, _ = rasterization(
            means=d["xyz"], quats=d["rotations"], scales=d["scales"],
            opacities=d["opacity"], colors=d["shs"],
            viewmats=train_cam.viewmatrix.unsqueeze(0),
            Ks=train_cam.K.unsqueeze(0),
            width=train_cam.image_width, height=train_cam.image_height,
            tile_size=args.tile_size, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )

        loss_data = canonical_combined_loss(train_img[0].clamp(0, 1), train_target, lambda_dssim=0.2)
        loss = loss_data["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        optim.step()

        # Measure the post-optimizer state.  Recompute activated values because
        # `d` was created before optim.step() and otherwise mixes stale derived
        # values (rotation/scale/opacity) with updated xyz parameter storage.
        d_eval = model.forward()
        # Render from FIXED camera 0 for measurement.
        with torch.no_grad():
            fixed_img, _, eval_info = rasterization(
                means=d_eval["xyz"], quats=d_eval["rotations"], scales=d_eval["scales"],
                opacities=d_eval["opacity"], colors=d_eval["shs"],
                viewmats=eval_cam.viewmatrix.unsqueeze(0),
                Ks=eval_cam.K.unsqueeze(0),
                width=eval_cam.image_width, height=eval_cam.image_height,
                tile_size=args.tile_size, packed=True, sh_degree=model.sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            )

        # Retain projection/3D state only for sampled t→t+1 membership pairs.
        capture_state = (
            step % args.membership_sample_interval == 0
            or (step + 1) % args.membership_sample_interval == 0
        )
        curr_data = extract_visible_data(eval_info, d_eval["xyz"], capture_state=capture_state)
        step_records[step] = curr_data
        n_gaussians_history.append(int(model.xyz.shape[0]))

        # Densification & pruning
        did_densify = False
        did_prune = False

        if step >= args.densify_start and step < 15000 and step % args.densify_interval == 0:
            e = model.densification(grad_threshold=2e-4, clone_max_screen_size=100.0,
                                    split_max_screen_size=100.0)
            if e["cloned"] + e["split"] > 0:
                did_densify = True

        if step >= args.prune_start and step % args.prune_interval == 0:
            n_pruned = model.prune_and_reset(opacity_threshold=0.005, reset_interval=3000,
                                              current_step=step)
            if n_pruned > 0:
                did_prune = True

        topology_changed = did_densify or did_prune
        if topology_changed:
            topology_events.append({
                "step": step,
                "densified": did_densify,
                "pruned": did_prune,
                "n_gaussians_before": n_gaussians_history[-2] if len(n_gaussians_history) > 1 else n_gaussians_history[-1],
                "n_gaussians_after": int(model.xyz.shape[0]),
            })
            optim = make_optim()

        # Compute overlap with previous step
        if step > 0:
            prev_data = step_records[step - 1]
            isect_metrics = compute_intersection_overlap(prev_data, curr_data)

            # Per-Gaussian membership: slow (CPU scatter), sample every 50 steps
            # A topology event is executed *after* snapshot t. Therefore it
            # affects the t→t+1 identity correspondence, not t−1→t.
            topology_since_previous_snapshot = any(
                event["step"] == step - 1 for event in topology_events
            )
            membership_metrics = {
                "n_common_gaussians": 0,
                "stable_membership_ratio": -1.0,
                "changed_membership_ratio": -1.0,
                "visibility_changed_ratio": -1.0,
                "position_change": None,
                "projected_center_change_px": None,
                "projected_radius_change_px": None,
                "membership_change_causes": None,
            }
            if (
                not topology_since_previous_snapshot
                and step % args.membership_sample_interval == 0
            ): 
                membership_metrics = compute_membership_stability_per_gaussian(prev_data, curr_data, args.tile_size)

            combined = {
                "step_t": step - 1,
                "step_t1": step,
                "topology_changed": topology_since_previous_snapshot,
                "n_gaussians_t": n_gaussians_history[-2],
                "n_gaussians_t1": n_gaussians_history[-1],
                **isect_metrics,
                **membership_metrics,
            }

            if topology_since_previous_snapshot:
                pair_metrics_topo.append(combined)
            else:
                pair_metrics_no_topo.append(combined)

            # Lag-2 (across no-topology span)
            if step >= 2 and not topology_since_previous_snapshot:
                prev2_data = step_records.get(step - 2)
                if prev2_data:
                    m = compute_intersection_overlap(prev2_data, curr_data)
                    pair_metrics_lag2.append({"step_t": step - 2, "step_t1": step, **m})

            # Lag-4 (across no-topology span)
            if step >= 4 and not topology_since_previous_snapshot:
                prev4_data = step_records.get(step - 4)
                if prev4_data:
                    had_topo = any(te["step"] in range(step - 3, step) for te in topology_events)
                    if not had_topo:
                        m = compute_intersection_overlap(prev4_data, curr_data)
                        pair_metrics_lag4.append({"step_t": step - 4, "step_t1": step, **m})

        # Keep only the t, t-2, and t-4 snapshots needed for the temporal
        # window.  Retaining all 500 full intersection tensors would consume
        # tens of GiB and corrupt the workload measurement through OOM pressure.
        old_step = step - 4
        if old_step >= 0:
            step_records.pop(old_step, None)

        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"[{step:04d}] N={model.xyz.shape[0]:,} "
                f"vis={curr_data['n_visible']:,} "
                f"isects={curr_data['n_entries']:,} "
                f"topo={topology_changed}",
                flush=True,
            )

    total_time = time.perf_counter() - t0

    # ── Aggregate ──
    def agg_pairs(pairs, label):
        if not pairs:
            return None
        reusable = [p["reusable_intersection_ratio"] for p in pairs]
        new_r = [p["new_intersection_ratio"] for p in pairs]
        removed = [p["removed_intersection_ratio"] for p in pairs]
        overlap = [p["intersection_overlap"] for p in pairs]
        stable = [p.get("stable_membership_ratio", -1.0) for p in pairs if p.get("stable_membership_ratio", -1.0) >= 0]
        changed = [p.get("changed_membership_ratio", -1.0) for p in pairs if p.get("changed_membership_ratio", -1.0) >= 0]
        result = {
            "reusable_intersection_ratio": stats(reusable),
            "new_intersection_ratio": stats(new_r),
            "removed_intersection_ratio": stats(removed),
            "intersection_overlap": stats(overlap),
            "stable_membership_ratio": stats(stable),
            "changed_membership_ratio": stats(changed),
            "n_pairs": len(pairs),
        }
        rp50 = result["reusable_intersection_ratio"]["p50"]
        changed_stats = result["changed_membership_ratio"]
        cp50_text = f"{changed_stats['p50']:.4f}" if changed_stats else "N/A"
        print(f"  [{label}] reusable(P50)={rp50:.4f} "
              f"changed(P50)={cp50_text} n={len(pairs)}")
        return result

    print(f"\n[C18] Total time: {total_time:.0f}s")
    print(f"[C18] No-topology pairs: {len(pair_metrics_no_topo)}")
    print(f"[C18] Topology pairs: {len(pair_metrics_topo)}")
    print(f"[C18] Lag-2 pairs: {len(pair_metrics_lag2)}")
    print(f"[C18] Lag-4 pairs: {len(pair_metrics_lag4)}")

    print("\n[C18] Aggregated statistics:")
    no_topo_agg = agg_pairs(pair_metrics_no_topo, "NO-TOPOLOGY")
    topo_agg = agg_pairs(pair_metrics_topo, "TOPOLOGY")
    lag2_agg = agg_pairs(pair_metrics_lag2, "LAG-2")
    lag4_agg = agg_pairs(pair_metrics_lag4, "LAG-4")

    # Phase analysis
    print("\n[C18] Phase analysis:")
    phases = {}
    for label, lo, hi in [("early", 0, 100), ("middle", 100, 400), ("late", 400, args.steps)]:
        subset = [r for r in pair_metrics_no_topo if lo <= r["step_t1"] < hi]
        phases[label] = agg_pairs(subset, f"Phase {label} ({lo}-{hi})")

    # Step-binned
    step_bins = {}
    for r in pair_metrics_no_topo:
        bin_idx = (r["step_t1"] // 100) * 100
        step_bins.setdefault(bin_idx, []).append(r)
    binned = {}
    for k, v in sorted(step_bins.items()):
        if v:
            binned[f"step_{k}_{k+99}"] = agg_pairs(v, f"BIN {k}")

    # ── Decision logic ──
    if no_topo_agg:
        p50_reusable = no_topo_agg["reusable_intersection_ratio"]["p50"]
        p50_changed = no_topo_agg["changed_membership_ratio"]["p50"]
        p50_stable = no_topo_agg["stable_membership_ratio"]["p50"]
        mean_stable = no_topo_agg["stable_membership_ratio"]["mean"]
    else:
        p50_reusable = 0.0
        p50_changed = 1.0
        p50_stable = 0.0
        mean_stable = 0.0

    if p50_changed < 0.10 and p50_reusable > 0.70:
        decision = "GO"
        decision_detail = (
            f"Strong: changed_membership_ratio P50={p50_changed:.2%} < 10% "
            f"AND reusable_intersection_ratio P50={p50_reusable:.2%} > 70%"
        )
    elif p50_changed < 0.30 and p50_reusable > 0.40:
        decision = "CONDITIONAL GO"
        decision_detail = (
            f"Conditional: changed={p50_changed:.2%} (threshold <30%), "
            f"reusable={p50_reusable:.2%} (threshold >40%)"
        )
    elif p50_changed < 0.50 and p50_reusable > 0.30:
        decision = "HIGH RISK"
        decision_detail = (
            f"High risk: changed={p50_changed:.2%} (threshold <50%), "
            f"reusable={p50_reusable:.2%} (threshold >30%)"
        )
    else:
        decision = "NO-GO"
        decision_detail = (
            f"changed={p50_changed:.2%} >= 50% OR "
            f"reusable={p50_reusable:.2%} < 40%"
        )

    # Topology impact
    tfrac = len(pair_metrics_topo) / max(len(pair_metrics_no_topo) + len(pair_metrics_topo), 1)
    topology_impact = {
        "n_events": len(topology_events),
        "fraction_of_steps": tfrac,
        "events_detail": topology_events,
    }
    if topo_agg:
        topology_impact["reusable_during_topo"] = topo_agg["reusable_intersection_ratio"]

    # ── Build JSON ──
    samples = pair_metrics_no_topo[:3] + (pair_metrics_no_topo[-1:] if len(pair_metrics_no_topo) > 3 else [])

    result = {
        "meta": {
            "title": "C18 Quantitative Decomposition Gate",
            "date": datetime.now(timezone.utc).isoformat(),
            "scene": args.scene,
            "steps": args.steps,
            "tile_size": args.tile_size,
            "resolution": args.resolution,
            "seed": args.seed,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gsplat_version": gsplat_version,
        },
        "decision": decision,
        "decision_detail": decision_detail,
        "decision_thresholds": {
            "strong_go": {"changed_membership_ratio_lt": 0.10, "reusable_intersection_ratio_gt": 0.70},
            "conditional_go": {"changed_membership_ratio_range": "10-30%", "reusable_intersection_ratio_range": "40-70%"},
            "high_risk": {"changed_membership_ratio_range": "30-50%", "reusable_intersection_ratio_range": "30-40%"},
            "no_go": {"changed_membership_ratio_gte": 0.50, "or_reusable_intersection_ratio_lt": 0.40},
        },
        "aggregated_no_topology": no_topo_agg,
        "aggregated_topology_pairs": topo_agg,
        "aggregated_lag2": lag2_agg,
        "aggregated_lag4": lag4_agg,
        "phase_analysis": phases,
        "binned_by_100_steps": binned,
        "topology_impact": topology_impact,
        "n_gaussians_history": n_gaussians_history,
        "total_training_time_s": total_time,
        "representative_samples": samples,
    }

    json_path = out / "c18_decomposition_gate.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[JSON] Saved: {json_path}")

    # ── Report ──
    report = generate_report(result, args)
    report_path = reports_dir / "c18_decomposition_gate.md"
    report_path.write_text(report)
    print(f"[Report] Saved: {report_path}")

    # ── Banner ──
    print(f"\n{'='*65}")
    print(f"  C18 DECOMPOSITION GATE — DECISION: {decision}")
    print(f"{'='*65}")
    if no_topo_agg:
        print(f"  Reusable intersection (P50): {p50_reusable:.4f}")
        print(f"  Changed membership (P50):    {p50_changed:.4f}")
        print(f"  Stable membership (P50):     {p50_stable:.4f}")
    print(f"  {decision_detail}")
    print(f"{'='*65}")

    return decision


def generate_report(result, args):
    lines = []
    lines.append("# C18 — Quantitative Decomposition Gate\n")
    lines.append(f"**Scene:** `{result['meta']['scene']}`  \n")
    lines.append(f"**Steps:** {result['meta']['steps']}  \n")
    lines.append(f"**Tile size:** {result['meta']['tile_size']}  \n")
    lines.append(f"**GPU:** {result['meta']['gpu']}  \n")
    lines.append(f"**gsplat:** {result['meta']['gsplat_version']}  \n")
    lines.append(f"**Date:** {result['meta']['date']}  \n")
    lines.append(f"**Fixed camera:** camera 0 (same viewpoint every step)\n")

    lines.append("---\n")
    lines.append(f"## Executive Decision: **{result['decision']}**\n")
    lines.append(f"{result['decision_detail']}\n")

    agg = result["aggregated_no_topology"]
    if agg:
        r = agg["reusable_intersection_ratio"]
        c = agg["changed_membership_ratio"]
        s = agg["stable_membership_ratio"]
        o = agg["intersection_overlap"]
        n = agg["new_intersection_ratio"]

        lines.append("\n## 1. Consecutive-Iteration Gaussian Stability\n")
        lines.append(f"### Intersection-level overlap (no-topology pairs, n={agg['n_pairs']})\n")
        lines.append("| Metric | P10 | P25 | P50 | P75 | P90 | Mean |\n")
        lines.append("|:-------|:--:|:--:|:--:|:--:|:--:|:----:|\n")
        lines.append(f"| Reusable intersection ratio | {r['p10']:.4f} | {r['p25']:.4f} | **{r['p50']:.4f}** | {r['p75']:.4f} | {r['p90']:.4f} | {r['mean']:.4f} |\n")
        lines.append(f"| Changed membership ratio | {c['p10']:.4f} | {c['p25']:.4f} | **{c['p50']:.4f}** | {c['p75']:.4f} | {c['p90']:.4f} | {c['mean']:.4f} |\n")
        lines.append(f"| Stable membership ratio | {s['p10']:.4f} | {s['p25']:.4f} | {s['p50']:.4f} | {s['p75']:.4f} | {s['p90']:.4f} | {s['mean']:.4f} |\n")
        lines.append(f"| Intersection overlap (Jaccard) | {o['p10']:.4f} | {o['p25']:.4f} | {o['p50']:.4f} | {o['p75']:.4f} | {o['p90']:.4f} | {o['mean']:.4f} |\n")
        lines.append(f"| New intersection ratio | {n['p10']:.4f} | {n['p25']:.4f} | {n['p50']:.4f} | {n['p75']:.4f} | {n['p90']:.4f} | {n['mean']:.4f} |\n")

    lines.append("\n### Temporal window\n")
    lines.append("| Lag | P50 reusable | P50 overlap | n_pairs |\n")
    lines.append("|:---:|:-----------:|:----------:|:-------:|\n")
    if agg:
        lines.append(f"| t→t+1 | {agg['reusable_intersection_ratio']['p50']:.4f} | {agg['intersection_overlap']['p50']:.4f} | {agg['n_pairs']} |\n")
    lag2 = result["aggregated_lag2"]
    lag4 = result["aggregated_lag4"]
    if lag2:
        lines.append(f"| t→t+2 | {lag2['reusable_intersection_ratio']['p50']:.4f} | {lag2['intersection_overlap']['p50']:.4f} | {lag2['n_pairs']} |\n")
    if lag4:
        lines.append(f"| t→t+4 | {lag4['reusable_intersection_ratio']['p50']:.4f} | {lag4['intersection_overlap']['p50']:.4f} | {lag4['n_pairs']} |\n")

    lines.append("\n## 2. Phase Analysis\n")
    for phase_name in ["early", "middle", "late"]:
        pd = result["phase_analysis"].get(phase_name)
        if pd:
            lines.append(f"- **{phase_name}**: reusable P50={pd['reusable_intersection_ratio']['p50']:.4f}, "
                        f"changed P50={pd['changed_membership_ratio']['p50']:.4f}, n={pd['n_pairs']}\n")

    lines.append("\n## 3. Topology-Change Impact\n")
    ti = result["topology_impact"]
    lines.append(f"- **Events:** {ti['n_events']} topology changes across {args.steps} steps\n")
    lines.append(f"- **Fraction:** {ti['fraction_of_steps']:.2%} of steps\n")
    if ti.get("reusable_during_topo"):
        tr = ti["reusable_during_topo"]
        lines.append(f"- **Reuse during topo steps:** P50={tr['p50']:.4f} "
                    f"(vs no-topo P50={agg['reusable_intersection_ratio']['p50']:.4f})\n")
    lines.append("- **Note:** Per-Gaussian membership comparison is INVALID for topology-change ")
    lines.append("pairs because pruning shifts Gaussian parameter indices. ")
    lines.append("Only intersection-level pair comparison is reported for those.\n")

    lines.append("\n## 4. Potential Pass2 Reduction\n")
    if agg:
        rp50 = agg["reusable_intersection_ratio"]["p50"]
        lines.append(f"- **Theoretical max Pass2 reduction:** {rp50:.2%} of intersection records (P50)\n")
        lines.append(f"- Range: P10={r['p10']:.2%} to P90={r['p90']:.2%}\n")
        est_records = 3_250_000
        lines.append(f"- Baseline records per step: ~{est_records:,}\n")
        lines.append(f"- Candidates for reuse: ~{int(est_records * rp50):,} records (P50)\n")
        lines.append(f"- Records requiring rebuild: ~{int(est_records * (1 - rp50)):,}\n")
        lines.append("**Caveat:** Theoretical upper bound. Actual speedup depends on invalidation ")
        lines.append("scan overhead, incremental merge cost, and memory management.\n")

    lines.append("\n## 5. Decision Criteria Check\n")
    thresholds = result["decision_thresholds"]
    lines.append("| Criterion | Threshold | Measured | Met? |\n")
    lines.append("|:----------|:---------:|:--------:|:----:|\n")
    if agg:
        rp50 = agg["reusable_intersection_ratio"]["p50"]
        cp50 = agg["changed_membership_ratio"]["p50"]
        lines.append(f"| Strong GO: changed membership | <{thresholds['strong_go']['changed_membership_ratio_lt']} | {cp50:.4f} | {'✅' if cp50 < 0.10 else '❌'} |\n")
        lines.append(f"| Strong GO: reusable intersect | >{thresholds['strong_go']['reusable_intersection_ratio_gt']} | {rp50:.4f} | {'✅' if rp50 > 0.70 else '❌'} |\n")
        lines.append(f"| Conditional: changed membership | {thresholds['conditional_go']['changed_membership_ratio_range']} | {cp50:.4f} | {'✅' if cp50 < 0.30 else '❌'} |\n")
        lines.append(f"| Conditional: reusable intersect | {thresholds['conditional_go']['reusable_intersection_ratio_range']} | {rp50:.4f} | {'✅' if rp50 > 0.40 else '❌'} |\n")
        if cp50 < 0.10 and rp50 > 0.70:
            lines.append("| **STRONG GO SATISFIED** | both | ✅ | ✅ |\n")
        elif cp50 < 0.30 and rp50 > 0.40:
            lines.append("| **CONDITIONAL GO SATISFIED** | both | ✅ | ✅ |\n")

    lines.append("\n### Verdict\n")
    lines.append(f"> **{result['decision']}** — {result['decision_detail']}\n")
    if result["decision"] == "NO-GO":
        lines.append("> **Action:** C18 stopped. Return to optimization-space exploration.\n")
    elif result["decision"] == "GO":
        lines.append("> **Action:** Proceed to CUDA prototype + forward correctness phase.\n")
    elif result["decision"] == "CONDITIONAL GO":
        lines.append("> **Action:** Proceed to CUDA prototype, but design must account for ")
        lines.append("partial reuse (implementation overhead must be carefully bounded).\n")
    elif result["decision"] == "HIGH RISK":
        lines.append("> **Action:** Requires strong implementation argument before CUDA work.\n")

    lines.append("\n---\n")
    lines.append("**Note:** Per-Gaussian membership stability measured only on no-topology pairs. ")
    lines.append("During densification/pruning, state must be fully rebuilt anyway.\n")

    return "".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C18 Decomposition Gate")
    ap.add_argument("--scene", default="room")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--tile-size", type=int, default=16)
    ap.add_argument("--resolution", default="1080p")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--densify-start", type=int, default=100)
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--prune-start", type=int, default=100)
    ap.add_argument("--prune-interval", type=int, default=100)
    ap.add_argument("--membership-sample-interval", type=int, default=50)
    args = ap.parse_args()

    decision = run_decomposition(args)
    sys.exit({"GO": 0, "CONDITIONAL GO": 0, "HIGH RISK": 1, "NO-GO": 2}.get(decision, 1))
