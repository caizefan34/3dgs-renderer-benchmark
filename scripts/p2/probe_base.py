#!/usr/bin/env python3
"""Probe how base flat forward is reachable: pip gsplat, core .so ops, C0 composed .so."""
import json, os, hashlib, sys

out = {}
# 1. What gsplat is installed
import subprocess
try:
    r = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if "gsplat" in l.lower() or "torch" in l.lower()]
    out["pip_gsplat_torch"] = lines
except Exception as e:
    out["pip_err"] = str(e)

# 2. gsplat package files
import importlib.util
spec = importlib.util.find_spec("gsplat")
out["gsplat_spec"] = None if spec is None else spec.origin
if spec is not None:
    import glob
    base = os.path.dirname(spec.origin)
    out["gsplat_dir"] = base
    for f in ("__init__.py", "rasterization/__init__.py", "csrc.so", "csrc/__init__.py"):
        p = os.path.join(base, f)
        if os.path.exists(p):
            out.setdefault("gsplat_files", {})[f] = hashlib.sha256(open(p,"rb").read()).hexdigest() if p.endswith(".so") else "exists"

# 3. Known core .so and composed base .so
candidates = {
    "core_authoritative": "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so",
    "core_c0_composed": "/mnt/storage_pool/liaoyuanjun/higs_c0_cache_composed/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
    "core_c0_default": "/mnt/storage_pool/liaoyuanjun/higs_c0_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
}
for k, p in candidates.items():
    if os.path.isfile(p):
        out.setdefault("so_candidates", {})[k] = {"size": os.path.getsize(p), "sha": hashlib.sha256(open(p,"rb").read()).hexdigest()}
    else:
        out.setdefault("so_missing", []).append((k, p))

print(json.dumps(out, sort_keys=True, indent=2))