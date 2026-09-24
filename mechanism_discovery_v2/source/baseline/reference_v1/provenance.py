"""
Reference V1 provenance — complete experiment provenance manifest.

Every run MUST generate provenance.json. If any required field is missing,
the run REFUSES TO START (fail-closed behavior).
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, Optional
from pathlib import Path


REQUIRED_FIELDS = [
    "experiment_id",
    "timestamp",
    "git_commit",
    "git_branch",
    "git_dirty",
    "training_script_path",
    "training_script_sha256",
    "gaussian_model_path",
    "gaussian_model_sha256",
    "config_path",
    "config_sha256",
    "renderer_version",
    "gsplat_version",
    "torch_version",
    "cuda_runtime",
    "gpu_name",
    "gpu_count",
    "scene",
    "scene_source_sha256",
    "initial_gaussian_count",
    "seed",
    "camera_seed",
    "optimizer_type",
    "optimizer_eps",
    "position_lr_init",
    "position_lr_final",
    "feature_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
    "densify_from_iter",
    "densify_until_iter",
    "densification_interval",
    "densify_grad_threshold",
    "percent_dense",
    "opacity_reset_interval",
    "min_opacity",
    "max_screen_size",
    "sh_degree",
    "sh_progress_interval",
    "lambda_dssim",
    "loss_type",
    "ssim_implementation",
    "command_line",
    "hostname",
]


def sha256_file(path: str) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def get_git_info(repo_root: str) -> Dict[str, str]:
    """Get git commit, branch, dirty status, diff sha256."""
    info = {"git_commit": "unknown", "git_branch": "unknown",
            "git_dirty": True, "git_diff_sha256": ""}

    def run_git(*args):
        try:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True, text=True, timeout=10,
                cwd=repo_root
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    commit = run_git("rev-parse", "HEAD")
    if commit:
        info["git_commit"] = commit
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        info["git_branch"] = branch
    status = run_git("status", "--porcelain")
    info["git_dirty"] = bool(status)
    if status:
        info["git_diff_sha256"] = sha256_string(status)
    return info


def get_gpu_info() -> Dict[str, str]:
    """Get GPU name and count."""
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        gpu_name, gpu_count = "N/A", 0
    return {"gpu_name": gpu_name, "gpu_count": gpu_count}


def get_software_versions() -> Dict[str, str]:
    """Get torch, cuda, gsplat versions."""
    info = {"torch_version": "unknown", "cuda_runtime": "unknown",
            "gsplat_version": "unknown", "renderer_version": "gsplat"}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda or "unknown"
    except Exception:
        pass
    try:
        import gsplat
        # gsplat 1.5.3 doesn't have __version__ — check pip
        result = subprocess.run(["pip3", "show", "gsplat"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    info["gsplat_version"] = line.split(":")[1].strip()
    except Exception:
        pass
    return info


def validate_provenance(provenance: dict, allow_dirty: bool = False) -> bool:
    """Validate that all required fields are present. Fail-closed.

    Returns True if valid, raises ValueError if not.
    """
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in provenance or provenance[field] is None:
            missing.append(field)

    if missing:
        raise ValueError(
            f"Provenance validation FAILED. Missing required fields:\n"
            + "\n".join(f"  - {f}" for f in missing)
        )

    if provenance.get("git_dirty", True) and not allow_dirty:
        raise ValueError(
            "Provenance validation FAILED: git_dirty=True. "
            "Commit changes before running paper experiments, "
            "or set allow_dirty=True for development runs."
        )

    return True


def build_provenance(
    config,
    training_script_path: str,
    gaussian_model_path: str,
    config_path: str,
    scene_source_path: str,
    initial_gaussian_count: int,
    camera_seed: int,
    allow_dirty: bool = False,
) -> dict:
    """Build complete provenance manifest.

    Args:
        config: ReferenceV1Config instance
        training_script_path: path to trainer.py
        gaussian_model_path: path to gaussian_model.py
        config_path: path to config file
        scene_source_path: path to PLY point cloud
        initial_gaussian_count: N after initialization
        camera_seed: seed used for camera sequence
        allow_dirty: if True, allow dirty git state (dev only)
    """
    repo_root = config.repo_root
    git_info = get_git_info(repo_root)
    gpu_info = get_gpu_info()
    sw_info = get_software_versions()

    provenance = {
        "experiment_id": f"ref_v1_{config.scene}_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_info["git_commit"],
        "git_branch": git_info["git_branch"],
        "git_dirty": git_info["git_dirty"],
        "git_diff_sha256": git_info["git_diff_sha256"],
        "training_script_path": training_script_path,
        "training_script_sha256": sha256_file(training_script_path),
        "gaussian_model_path": gaussian_model_path,
        "gaussian_model_sha256": sha256_file(gaussian_model_path),
        "config_path": config_path,
        "config_sha256": sha256_file(config_path) if os.path.exists(config_path) else "",
        "renderer_version": sw_info["renderer_version"],
        "gsplat_version": sw_info["gsplat_version"],
        "torch_version": sw_info["torch_version"],
        "cuda_runtime": sw_info["cuda_runtime"],
        "gpu_name": gpu_info["gpu_name"],
        "gpu_count": gpu_info["gpu_count"],
        "scene": config.scene,
        "scene_source_sha256": sha256_file(scene_source_path),
        "initial_gaussian_count": initial_gaussian_count,
        "seed": config.seed,
        "camera_seed": camera_seed,
        "optimizer_type": "Adam",
        "optimizer_eps": 1e-15,
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "densify_from_iter": config.densify_from_iter,
        "densify_until_iter": config.densify_until_iter,
        "densification_interval": config.densification_interval,
        "densify_grad_threshold": config.densify_grad_threshold,
        "percent_dense": config.percent_dense,
        "opacity_reset_interval": config.opacity_reset_interval,
        "min_opacity": config.min_opacity,
        "max_screen_size": config.max_screen_size,
        "sh_degree": config.sh_degree,
        "sh_progress_interval": config.sh_progress_interval,
        "lambda_dssim": config.lambda_dssim,
        "loss_type": "L1 + D-SSIM",
        "ssim_implementation": "separable_ssim",
        "command_line": " ".join(sys.argv),
        "hostname": os.uname().nodename if hasattr(os, 'uname') else os.environ.get("HOSTNAME", "unknown"),
        "reference_commit": "54c035f7834b564019656c3e3fcc3646292f727d",
    }

    validate_provenance(provenance, allow_dirty=allow_dirty)
    return provenance
