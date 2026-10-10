#!/usr/bin/env python3
"""S1A environment freeze: record source/binary identity, loaded paths, GPU, disk."""
import os, sys, glob, hashlib, json, subprocess, platform, time

def sha256_file(p):
    if not os.path.exists(p): return None
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def git_commit(repo):
    try:
        return subprocess.run(["git","-C",repo,"rev-parse","HEAD"],
                              capture_output=True,text=True).stdout.strip()
    except Exception: return None

out = {}
out["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
out["host"] = subprocess.run(["hostname"],capture_output=True,text=True).stdout.strip()

# Source + binary identities
src = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
so = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/gsplat_scene_cuda.so"
out["source"] = {"path":src,"commit":git_commit(src),
    "gaussian_inference_sha256":sha256_file(f"{src}/gsplat/experimental/render/functional/gaussian_inference.py"),
    "cuda_wrapper_sha256":sha256_file(f"{src}/gsplat/cuda/_wrapper.py"),
    "rendering_sha256":sha256_file(f"{src}/gsplat/rendering.py")}
out["binary"] = {"path":so,"sha256":sha256_file(so),"size_bytes":os.path.getsize(so)}

# torch / cuda / python
import torch
out["runtime"] = {"python":platform.python_version(),"torch":torch.__version__,
    "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cc":list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}

# GPU occupancy snapshot
gpu = subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                      "--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
out["gpu_snapshot"] = gpu
procs = subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
out["gpu_processes"] = procs

# Disk
disk = subprocess.run(["df","-h","/","/mnt/storage_pool"],capture_output=True,text=True).stdout.strip()
out["disk"] = disk

print(json.dumps(out, indent=2))
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_ENV_FREEZE.json","w") as f:
    json.dump(out, f, indent=2)
print("\nsaved S1A_ENV_FREEZE.json")
