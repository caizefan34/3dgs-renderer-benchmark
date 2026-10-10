#!/usr/bin/env python3
"""Collect SHA256 identities for source, binaries, checkpoints, and cameras."""
import hashlib, json, os, subprocess, glob

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def git_commit(repo):
    try:
        return subprocess.run(["git","-C",repo,"rev-parse","HEAD"],
                              capture_output=True,text=True).stdout.strip()
    except Exception:
        return None

out = {}

# Source
src = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
out["source"] = {
    "path": src,
    "commit": git_commit(src),
    "description": "HiGS patched gsplat source; commit 77ab983 = gsplat upstream base + HiGS patches (dirty worktree)",
    "experimental_init_sha256": sha256_file(f"{src}/gsplat/experimental/__init__.py"),
    "gaussian_inference_sha256": sha256_file(f"{src}/gsplat/experimental/render/functional/gaussian_inference.py"),
    "rendering_sha256": sha256_file(f"{src}/gsplat/rendering.py"),
    "cuda_wrapper_sha256": sha256_file(f"{src}/gsplat/cuda/_wrapper.py"),
}

# Binaries
so = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/gsplat_scene_cuda.so"
out["binary"] = {
    "path": so,
    "sha256": sha256_file(so),
    "size_bytes": os.path.getsize(so),
    "built_from": "commit 77ab983 (gsplat-higs-mx source)",
    "build_env": "higs-13scene-env: Python 3.10.21, torch 2.9.1+cu128, CUDA 12.8, nvcc 12.8, sm_80",
    "exports": ["pack_gaussian_inference_scene"],
    "build_params_sha256": sha256_file("/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/build_params.json"),
}

# Main gsplat csrc.so (from ~/.local — the render inference op backend)
main_so_candidates = glob.glob("/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat/csrc.so")
out["main_csrc_so"] = []
for p in main_so_candidates:
    out["main_csrc_so"].append({"path": p, "sha256": sha256_file(p), "size_bytes": os.path.getsize(p)})

# Checkpoints (6)
ckpt_base = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused"
ckpts = []
for scene, run in [("garden","fp1_garden_2026-10-06-01-32-48"),("bicycle","fp1_bicycle_2026-10-06-01-01-33")]:
    for f in ["0005000.pt","0010000.pt","final.pt"]:
        p = f"{ckpt_base}/{run}/checkpoints/{f}"
        if os.path.exists(p):
            ckpts.append({"scene":scene,"file":f,"path":p,"sha256":sha256_file(p),"size_bytes":os.path.getsize(p)})
out["checkpoints"] = ckpts

# Camera files
cams = []
for scene in ["garden","bicycle"]:
    p = f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/{scene}/eval_cameras.json"
    if os.path.exists(p):
        cams.append({"scene":scene,"path":p,"sha256":sha256_file(p)})
out["cameras"] = cams

with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S0R2_IDENTITIES.json","w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:2000])
print("\n... saved full to S0R2_IDENTITIES.json")
