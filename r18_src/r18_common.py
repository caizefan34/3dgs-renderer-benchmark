#!/usr/bin/env python3
"""R18 shared utilities — validated benchmark: correctness + performance, std_ll vs HiGS.

Extends the R17 common module with:
  - 1080p camera loading + render grid
  - correctness comparison helpers (MAE / RMSE / max abs error, shape checks)
  - device-side GPU utilization sampling
  - peak reserved + peak allocated memory snapshots

All defaults target the MX machine. Every path can be overridden by env vars.
Nothing here modifies renderer sources, binaries or checkpoints.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Expected identities (from R17 handoff / preflight records)
# ----------------------------------------------------------------------------
EXPECTED = {
    "ckpt_path": "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt",
    "ckpt_sha256": "ccec425c8da02d423fb66e89450e3c837209f280deea24dbf4a507bc3cd4ea14",
    "n_gaussians": 4_413_928,
    "higs_csrc_sha256": "b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3",
    "std_csrc_sha256": "40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50",
    "subset_n": 580_000,
    "subset_seed": 42,
}

PROJECT = Path(os.environ.get(
    "HIGS_PROJECT",
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"))
OUT = Path(os.environ.get(
    "R18_OUT", "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_VALIDATED_BENCHMARK_R18_MX"))
CKPT = Path(os.environ.get("R18_CKPT", EXPECTED["ckpt_path"]))

CAMERA_JSON_CANDIDATES = [
    os.environ.get("R18_CAMERAS_JSON", ""),
    "/home/liaoyuanjun/higs-13scene/data/official/mipnerf360/bicycle/cameras.json",
    "/home/liaoyuanjun/3dgs-renderer-benchmark/data/official/mipnerf360/bicycle/cameras.json",
    "/mnt/storage_pool/liaoyuanjun/3dgs-renderer-benchmark/data/official/mipnerf360/bicycle/cameras.json",
]

# Render grids
W540, H540 = 960, 540
W1080, H1080 = 1920, 1080
TILE_SIZE = 16


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
    for sub in ("", "logs", "raw", "captures"):
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


def full_indices(n_full: int) -> np.ndarray:
    return np.arange(n_full, dtype=np.int64)


# ----------------------------------------------------------------------------
# Activation contract (training-source semantics) — identical to R17
# ----------------------------------------------------------------------------
def act_opacity(raw: torch.Tensor) -> torch.Tensor:
    out = torch.sigmoid(raw)
    return out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out


def act_scales(raw: torch.Tensor) -> torch.Tensor:
    return torch.exp(raw)


def act_quats(raw: torch.Tensor) -> torch.Tensor:
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
# Real training camera protocol — identical to R17
# ----------------------------------------------------------------------------
def real_camera_from_3dgs_json(cd: dict, width: int, height: int):
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


# ----------------------------------------------------------------------------
# Correctness comparison helpers
# ----------------------------------------------------------------------------
def _as_cpu_float(t):
    """Always return a 1-D float64 numpy array for metric math."""
    if torch.is_tensor(t):
        arr = t.detach().to(torch.float64).cpu().numpy()
    else:
        arr = np.asarray(t, dtype=np.float64)
    return np.ascontiguousarray(arr).reshape(-1)


def err_metrics(a, b) -> dict:
    """a, b: same-shape tensors. Returns MAE / RMSE / max-abs / mean / rel-L2."""
    x = _as_cpu_float(a).reshape(-1)
    y = _as_cpu_float(b).reshape(-1)
    if x.shape != y.shape:
        return {"shape_mismatch": True, "shape_a": list(np.shape(_as_cpu_float(a))),
                "shape_b": list(np.shape(_as_cpu_float(b)))}
    d = x - y
    denom = float(np.linalg.norm(y)) + 1e-12
    return {
        "shape_mismatch": False,
        "n": int(x.size),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))) if x.size else 0.0,
        "mean_signed": float(np.mean(d)),
        "rel_l2": float(np.linalg.norm(d) / denom),
        "a_absmax": float(np.max(np.abs(x))) if x.size else 0.0,
        "b_absmax": float(np.max(np.abs(y))) if y.size else 0.0,
    }


def grad_metrics(ga, gb) -> dict:
    """Per-tensor gradient comparison. ga, gb same-shape tensors (may be None)."""
    if ga is None and gb is None:
        return {"both_none": True}
    if ga is None or gb is None:
        return {"one_none": True, "a_none": ga is None, "b_none": gb is None}
    x = _as_cpu_float(ga).reshape(-1)
    y = _as_cpu_float(gb).reshape(-1)
    if x.shape != y.shape:
        return {"shape_mismatch": True, "shape_a": list(x.shape), "shape_b": list(y.shape)}
    d = x - y
    denom = float(np.linalg.norm(y)) + 1e-12
    return {
        "shape_mismatch": False,
        "n": int(x.size),
        "cosine": float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))) if x.size else 0.0,
        "rel_l2": float(np.linalg.norm(d) / denom),
        "a_absmax": float(np.max(np.abs(x))) if x.size else 0.0,
        "b_absmax": float(np.max(np.abs(y))) if y.size else 0.0,
        "a_l2": float(np.linalg.norm(x)),
        "b_l2": float(np.linalg.norm(y)),
    }


def zero_grad_frac(g) -> dict:
    if g is None:
        return {"none": True}
    x = _as_cpu_float(g).reshape(-1)
    n = int(x.size)
    nz = int(torch.count_nonzero(torch.from_numpy(x).to(torch.float32)).item()) if n else 0
    return {"n": n, "n_zero": n - nz, "frac_zero": (n - nz) / n if n else 0.0,
            "frac_nonzero": nz / n if n else 0.0}


# ----------------------------------------------------------------------------
# Memory / device stats
# ----------------------------------------------------------------------------
def gpu_mem_snapshot(tag: str = "") -> dict:
    if not torch.cuda.is_available():
        return {"tag": tag, "cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "tag": tag, "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "free_gb": free / 1e9, "total_gb": total / 1e9,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def device_utilization_ms(window_ms: float = 100.0) -> dict:
    """Sample nvidia-smi utilization over a short window (best-effort, host-side)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        return {"gpu_util_pct": float(parts[0]) if parts else None,
                "mem_used_mib": float(parts[1]) if len(parts) > 1 else None,
                "mem_total_mib": float(parts[2]) if len(parts) > 2 else None,
                "note": "instantaneous sample at end of timed loop"}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def read_common_constants(project: Path = PROJECT):
    from oracle_higs import parse_value
    header = Path(project) / "gsplat/cuda/include/Common.h"
    if not header.is_file():
        return {"header": str(header), "found": False}
    text = header.read_text(errors="replace")
    return {
        "header": str(header), "sha256": sha256_file(header),
        "alpha_threshold": parse_value(text, "ALPHA_THRESHOLD"),
        "gaussian_extend": parse_value(text, "GAUSSIAN_EXTEND"),
        "found": True,
    }
