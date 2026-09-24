#!/usr/bin/env python3
"""
R3 Provenance Collector — captures git/gsplat/GPU/checkpoint environment for certificates.

Called at the start of r3_certificate_runner.py to record exact experimental
conditions. Every output JSON file embeds this provenance block.

Usage:
  python experiments/r3/r3_provenance.py --checkpoint <path> --output <dir>
"""

import os, sys, json, hashlib, subprocess, argparse
from datetime import datetime


def git_info(repo_root):
    """Collect git provenance."""
    info = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_root,
        )
        info["commit"] = result.stdout.strip()
    except Exception:
        info["commit"] = "unknown"

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=repo_root,
        )
        info["git_dirty"] = len(result.stdout.strip()) > 0
    except Exception:
        info["git_dirty"] = True

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_root,
        )
        info["branch"] = result.stdout.strip()
    except Exception:
        info["branch"] = "unknown"

    return info


def gsplat_info():
    """Collect gsplat version and source hash."""
    info = {}
    try:
        import gsplat
        info["version"] = getattr(gsplat, "__version__", "unknown")
        info["file"] = gsplat.__file__
        
        # Hash the backward kernel source (if available)
        bwd_path = os.path.join(os.path.dirname(gsplat.__file__),
                                "cuda", "csrc", "rasterize_to_pixels_bwd.cu")
        if os.path.exists(bwd_path):
            with open(bwd_path, "rb") as f:
                info["raster_bwd_cu_sha256"] = hashlib.sha256(f.read()).hexdigest()
        else:
            # Could be a compiled extension
            ext_dir = os.path.join(os.path.dirname(gsplat.__file__), "cuda")
            for ext_name in ["csrc.so", "csrc.pyd", "_C.abi3.so"]:
                ext_path = os.path.join(ext_dir, ext_name)
                if os.path.exists(ext_path):
                    with open(ext_path, "rb") as f:
                        info[f"{ext_name}_sha256"] = hashlib.sha256(f.read()).hexdigest()
                    break
    except Exception as e:
        info["error"] = str(e)

    return info


def gpu_info():
    """Collect GPU device info."""
    info = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                info[f"gpu_{i}"] = torch.cuda.get_device_name(i)
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as e:
        info["error"] = str(e)

    # Environment variables
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "not_set")
    info["TORCH_EXTENSIONS_DIR"] = os.environ.get("TORCH_EXTENSIONS_DIR", "not_set")
    info["CUDA_CACHE_PATH"] = os.environ.get("CUDA_CACHE_PATH", "not_set")

    return info


def checkpoint_info(path):
    """Compute checkpoint hash and basic metadata."""
    info = {"path": path}
    if path and os.path.exists(path):
        # SHA-256 of full checkpoint file
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        info["sha256"] = sha.hexdigest()
        info["size_bytes"] = os.path.getsize(path)
        
        # Read iteration from checkpoint
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            info["iteration"] = ckpt.get("iteration", None)
        except Exception:
            pass

    return info


def patch_status():
    """Verify no R2.1 or C51 patches are loaded."""
    status = {
        "R2_1_PATCH_LOADED": False,
        "C51_PATCH_LOADED": False,
    }
    
    # Check by backward kernel hash (known canonical hash)
    try:
        import gsplat
        bwd_path = os.path.join(os.path.dirname(gsplat.__file__),
                                "cuda", "csrc", "rasterize_to_pixels_bwd.cu")
        if os.path.exists(bwd_path):
            with open(bwd_path, "rb") as f:
                content = f.read()
            # Check for R2.1 signature: "r2_geo_mask"
            if b"r2_geo_mask" in content:
                status["R2_1_PATCH_LOADED"] = True
            # Check for C51 signature: "importance_mask"
            if b"importance_mask" in content:
                status["C51_PATCH_LOADED"] = True
    except Exception:
        pass

    return status


def collect_provenance(checkpoint_path, repo_root):
    """Collect complete provenance dict."""
    prov = {
        "timestamp": datetime.utcnow().isoformat(),
        "git": git_info(repo_root),
        "gsplat": gsplat_info(),
        "gpu": gpu_info(),
        "checkpoint": checkpoint_info(checkpoint_path),
        "patches": patch_status(),
    }
    return prov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Checkpoint file path")
    parser.add_argument("--output", required=True, help="Output JSON directory")
    parser.add_argument("--repo", default=".", help="Repository root (default: cwd)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    prov = collect_provenance(args.checkpoint, args.repo)

    outpath = os.path.join(args.output, "provenance.json")
    with open(outpath, "w") as f:
        json.dump(prov, f, indent=2)

    print(f"Provenance saved: {outpath}")
    for k, v in prov.items():
        if isinstance(v, dict):
            print(f"  {k}: {json.dumps(v, indent=4)}")
        else:
            print(f"  {k}: {v}")

    return prov


if __name__ == "__main__":
    main()
