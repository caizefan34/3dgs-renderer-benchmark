#!/usr/bin/env python3
"""Probe the A100 environment for AccuTile validation."""
import sys, json, subprocess, os

info = {}
try:
    import gsplat
    info["gsplat_version"] = gsplat.__version__
    info["gsplat_file"] = gsplat.__file__
except Exception as e:
    info["gsplat_error"] = str(e)

try:
    import torch
    info["torch_version"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["gpu_count"] = torch.cuda.device_count()
    info["compute_cap"] = torch.cuda.get_device_capability(0)
except Exception as e:
    info["torch_error"] = str(e)

info["python"] = sys.version.split()[0]
info["executable"] = sys.executable

# Check nvcc
try:
    r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        for line in r.stdout.strip().split("\n"):
            if "release" in line.lower():
                info["nvcc"] = line.strip()
except Exception:
    pass

# Check gcc
try:
    r = subprocess.run(["gcc", "--version"], capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        info["gcc"] = r.stdout.strip().split("\n")[0]
except Exception:
    pass

print(json.dumps(info, indent=2))
