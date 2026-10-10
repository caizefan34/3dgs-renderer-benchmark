#!/usr/bin/env python3
"""Generate SHA256 manifest of all S0R2 deliverables + key artifacts."""
import hashlib, json, os, glob

BASE = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
LOCAL = "C:/Users/36570/3dgs-renderer-benchmark/WORK_GRADIENT_ORACLE0_MX/WORK_GRADIENT_ORACLE0_MX"

def sha256_file(p):
    if not os.path.exists(p): return None
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

# Deliverables produced locally (need upload to MX for hashing) OR produced on MX.
# Hash the MX-side artifacts we generated during the run.
mx_artifacts = {
    "S0R2_IDENTITIES.json": f"{BASE}/S0R2_IDENTITIES.json",
    "S0R2_SMOKE_RESULT.json": f"{BASE}/S0R2_SMOKE_RESULT.json",
    "S0R2_REAL_CKPT_RESULT.json": f"{BASE}/S0R2_REAL_CKPT_RESULT.json",
    "gsplat_scene_cuda.so": "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/gsplat_scene_cuda.so",
    "gsplat_scene_cuda/build_params.json": "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/build_params.json",
    "source/gsplat/experimental/render/functional/gaussian_inference.py": "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/experimental/render/functional/gaussian_inference.py",
    "source/gsplat/cuda/_wrapper.py": "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/cuda/_wrapper.py",
    "source/gsplat/rendering.py": "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/rendering.py",
    "run_higs_train_benchmark.py": f"{BASE}/run_higs_train_benchmark.py",
    "bicycle_eval_cameras.json": "/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json",
    "bicycle_training_config.yaml": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/training_config.yaml",
    "bicycle_0005000.pt": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt",
}

# Also list the local deliverable files (report + JSONs) by their local paths
local_deliverables = [
    "S0R2_SOURCE_IDENTITY.json","S0R2_BUILD_IDENTITY.json","S0R2_RUNTIME_ABI.json",
    "S0R2_CHECKPOINT_COMPATIBILITY.json","S0R2_SPARSE_SMOKE.json","S0R2_REPORT.md","RUN_STATUS.json",
    "SHA256_MANIFEST.json"
]

manifest = {"mx_side_artifacts": {}, "local_deliverables": {}}
for k,p in mx_artifacts.items():
    manifest["mx_side_artifacts"][k] = {"path":p,"sha256":sha256_file(p),
        "size_bytes": (os.path.getsize(p) if os.path.exists(p) else None)}

import os
for f in local_deliverables:
    p = os.path.join(LOCAL, f)
    manifest["local_deliverables"][f] = {"path":p,"sha256":sha256_file(p),
        "size_bytes": (os.path.getsize(p) if os.path.exists(p) else None)}

with open(f"{BASE}/SHA256_MANIFEST.json","w") as fp:
    json.dump(manifest, fp, indent=2)
print(json.dumps(manifest, indent=2)[:1500])
print("\nsaved SHA256_MANIFEST.json")
