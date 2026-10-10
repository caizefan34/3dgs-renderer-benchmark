#!/usr/bin/env python3
"""R17 shared utilities — paths, checkpoint, activation contract, real cameras, stats.

All defaults target the MX machine. Every path can be overridden by environment
variables so the package stays executable from any directory layout.

Nothing in this package modifies renderer sources, binaries or checkpoints.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Expected identities (from R3/R8/R15/R16 handoff records)
# ----------------------------------------------------------------------------
EXPECTED = {
    "ckpt_path": "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt",
    "ckpt_sha256": "ccec425c8da02d423fb66e89450e3c837209f280deea24dbf4a507bc3cd4ea14",
    "n_gaussians": 4_413_928,
    "higs_csrc_sha256": "b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3",
    "std_csrc_sha256": "40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50",
    "subset_n": 580_000,
    "subset_seed": 42,
    "sample_seed": 42,
    "sample_k": 1000,
}

PROJECT = Path(os.environ.get(
    "HIGS_PROJECT",
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"))
OUT = Path(os.environ.get(
    "R17_OUT",
    "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX"))
CKPT = Path(os.environ.get("R17_CKPT", EXPECTED["ckpt_path"]))
R15_CAPTURE = Path(os.environ.get(
    "R15_CAPTURE",
    "/mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX/captured_projected_inputs.pt"))
R16_DIR = Path(os.environ.get(
    "R16_OUT", "/mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX"))

CAMERA_JSON_CANDIDATES = [
    os.environ.get("R17_CAMERAS_JSON", ""),
    str(Path(os.environ.get("TRAIN_REPO", str(Path.home() / "3dgs-renderer-benchmark")))
        / "data/official/mipnerf360/bicycle/cameras.json"),
    "/home/liaoyuanjun/3dgs-renderer-benchmark/data/official/mipnerf360/bicycle/cameras.json",
    "/mnt/storage_pool/liaoyuanjun/3dgs-renderer-benchmark/data/official/mipnerf360/bicycle/cameras.json",
]

# 540p render grid used by R3/R8 (HN-540) and R17 P3/P4
W540, H540 = 960, 540
TILE_SIZE = 16
W_TILES_540 = W540 // TILE_SIZE      # 60
H_TILES_540 = H540 // TILE_SIZE      # 34
TILES_540 = W_TILES_540 * H_TILES_540  # 2040


def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_out_dirs(out: Path = OUT):
    for sub in ("", "logs", "raw", "oracle", "captures"):
        (out / sub if sub else out).mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"[write] {path}", flush=True)
    return path


def append_csv(path, row, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def add_project_to_syspath(project: Path = PROJECT):
    project = str(project)
    if project not in sys.path:
        sys.path.insert(0, project)


def resolve_cameras_json() -> str | None:
    for cand in CAMERA_JSON_CANDIDATES:
        if cand and Path(cand).is_file():
            return cand
    return None


# ----------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------
def load_checkpoint(path: Path = CKPT, require: bool = True):
    info = {"path": str(path), "exists": Path(path).is_file(),
            "sha256": sha256_file(path)}
    if not Path(path).is_file():
        if require:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return None, info
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    info["n_gaussians"] = int(ckpt["xyz"].shape[0])
    info["keys"] = sorted(ckpt.keys())
    return ckpt, info


def build_580k_indices(n_full: int) -> np.ndarray:
    """Exact R8/R15 580K protocol: np.random.seed(42); sorted choice, no replace."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(EXPECTED["subset_seed"])
        idx = np.sort(np.random.choice(n_full, EXPECTED["subset_n"], replace=False))
    finally:
        np.random.set_state(rng_state)
    return idx


def choose_ids(n: int, k: int, seed: int) -> np.ndarray:
    """Exact R15/R16 sample protocol (default_rng(seed).choice, sorted)."""
    k = min(n, k)
    if k == n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


# ----------------------------------------------------------------------------
# Activation contract (training-source semantics)
# ----------------------------------------------------------------------------
def act_opacity(raw: torch.Tensor) -> torch.Tensor:
    """get_opacity(): torch.sigmoid(self._opacity).squeeze(-1)"""
    out = torch.sigmoid(raw)
    return out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out


def act_scales(raw: torch.Tensor) -> torch.Tensor:
    """get_scaling(): torch.exp(self._scaling)"""
    return torch.exp(raw)


def act_quats(raw: torch.Tensor) -> torch.Tensor:
    """get_rotation(): F.normalize(self._rotation, dim=-1)"""
    return torch.nn.functional.normalize(raw, dim=-1)


def build_cell_state(ckpt, indices, opacity_mode: str, scale_mode: str,
                     quat_mode: str, device="cpu"):
    """opacity_mode in {raw, sigmoid}; scale_mode in {raw, exp}; quat_mode in {raw, normalize}."""
    xyz = ckpt["xyz"][indices]
    shs = ckpt["shs"][indices]
    scaling = ckpt["scaling"][indices]
    rotation = ckpt["rotation"][indices]
    opacity = ckpt["opacity"][indices]
    if opacity_mode == "sigmoid":
        opacity = torch.sigmoid(opacity)
    if opacity.dim() > 1 and opacity.shape[-1] == 1:
        opacity = opacity.squeeze(-1)
    if scale_mode == "exp":
        scaling = torch.exp(scaling)
    if quat_mode == "normalize":
        rotation = act_quats(rotation)
    state = {
        "means": xyz.to(device).contiguous(),
        "quats": rotation.to(device).contiguous(),
        "scales": scaling.to(device).contiguous(),
        "opacities": opacity.to(device).contiguous(),
        "shs": shs.to(device).contiguous(),
    }
    if device != "cpu":
        for k in state:
            state[k].requires_grad_(True)
    return state


# ----------------------------------------------------------------------------
# Real training camera protocol (replicated from benchmark_framework + GTDataset)
# ----------------------------------------------------------------------------
def real_camera_from_3dgs_json(cd: dict, width: int, height: int):
    """_camera_from_3dgs_json + resize_cameras, exactly as the training stack.

    Original mipnerf360 json entry: width/height/fx/fy/position/rotation(c2w 3x3).
    Resize preserves tanfov = orig_W/(2*fx); principal point recentered.
    """
    ow, oh = int(cd["width"]), int(cd["height"])
    fx, fy = float(cd["fx"]), float(cd["fy"])
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = np.asarray(cd["rotation"], dtype=np.float32)
    cam2world[:3, 3] = np.asarray(cd["position"], dtype=np.float32)
    viewmatrix = torch.tensor(np.linalg.inv(cam2world).astype(np.float32))
    tanfovx = ow / (2.0 * fx)
    tanfovy = oh / (2.0 * fy)
    K = torch.tensor([
        [width / (2.0 * tanfovx), 0.0, width / 2.0],
        [0.0, height / (2.0 * tanfovy), height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    return {
        "K": K, "viewmatrix": viewmatrix,
        "camera_center": torch.tensor(cd["position"], dtype=torch.float32),
        "tanfovx": tanfovx, "tanfovy": tanfovy,
        "orig_width": ow, "orig_height": oh,
        "img_name": cd.get("img_name"), "id": cd.get("id"),
        "width": width, "height": height,
    }


def load_real_cameras(width: int = W540, height: int = H540, device="cpu"):
    """Returns (list_of_camera_dicts, source_json_path, sha256). Raises if missing."""
    path = resolve_cameras_json()
    if path is None:
        raise FileNotFoundError(
            "bicycle cameras.json not found in any candidate location: "
            + " | ".join(c for c in CAMERA_JSON_CANDIDATES if c))
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected cameras.json layout at {path}")
    cams = []
    for cd in data:
        cam = real_camera_from_3dgs_json(cd, width, height)
        cam["K"] = cam["K"].to(device)
        cam["viewmatrix"] = cam["viewmatrix"].to(device)
        cam["camera_center"] = cam["camera_center"].to(device)
        cams.append(cam)
    return cams, path, sha256_file(path)


def camera_depth_stats(means_cpu: torch.Tensor, viewmatrix: torch.Tensor) -> dict:
    """Fraction of gaussians with positive camera-space depth (CPU math)."""
    m = means_cpu.float()
    ones = torch.ones(m.shape[0], 1)
    hom = torch.cat([m, ones], dim=1)
    cam = hom @ viewmatrix.T  # [N,4]
    z = cam[:, 2]
    return {
        "n": int(z.numel()),
        "frac_positive_depth": float((z > 0).float().mean().item()),
        "min_depth": float(z.min().item()),
        "median_depth": float(z.median().item()),
        "max_depth": float(z.max().item()),
    }


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def np_stats(x, prefix: str = "", quantiles=(0.5, 0.9, 0.99)) -> dict:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {f"{prefix}n": 0}
    out = {
        f"{prefix}n": int(v.size),
        f"{prefix}mean": float(np.mean(v)),
        f"{prefix}min": float(np.min(v)),
        f"{prefix}max": float(np.max(v)),
    }
    for q in quantiles:
        out[f"{prefix}p{int(q*100)}"] = float(np.quantile(v, q))
    return out


def tensor_cpu(x):
    return x.detach().cpu().contiguous() if torch.is_tensor(x) else x


def nan_report(*named_tensors) -> dict:
    out = {}
    for name, t in named_tensors:
        if t is None:
            out[name] = "NONE"
            continue
        tt = t.detach()
        out[name] = {
            "shape": list(tt.shape),
            "has_nan": bool(torch.isnan(tt).any().item()),
            "has_inf": bool(torch.isinf(tt).any().item()),
        }
    return out


def gpu_mem_snapshot(tag: str = "") -> dict:
    if not torch.cuda.is_available():
        return {"tag": tag, "cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "tag": tag,
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "free_gb": free / 1e9,
        "total_gb": total / 1e9,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def read_common_constants(project: Path = PROJECT):
    """Read actual ALPHA_THRESHOLD / GAUSSIAN_EXTEND from the frozen source tree."""
    from oracle_higs import parse_value
    header = Path(project) / "gsplat/cuda/include/Common.h"
    if not header.is_file():
        return {"header": str(header), "found": False}
    text = header.read_text(errors="replace")
    return {
        "header": str(header),
        "sha256": sha256_file(header),
        "alpha_threshold": parse_value(text, "ALPHA_THRESHOLD"),
        "gaussian_extend": parse_value(text, "GAUSSIAN_EXTEND"),
        "found": True,
    }
