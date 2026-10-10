#!/usr/bin/env python3
"""Probe a candidate Python env for HiGS gsplat compatibility. Read-only."""
import sys, os, json, hashlib

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

print("=== ENV PROBE ===")
print("python_exe:", sys.executable)
print("python_ver:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("cuda_avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
    print("torch_file:", torch.__file__)
except Exception as e:
    print("TORCH_FAIL:", e)

# gsplat
try:
    import gsplat
    print("gsplat:", getattr(gsplat, "__version__", "?"))
    print("gsplat_file:", gsplat.__file__)
    try:
        from gsplat import experimental
        print("gsplat.experimental: OK")
        print("experimental_file:", experimental.__file__)
        has_dyn = hasattr(experimental, "rasterize_gaussian_higs_dynamic")
        print("has_rasterize_gaussian_higs_dynamic:", has_dyn)
    except Exception as e:
        print("gsplat.experimental: FAIL", e)
except Exception as e:
    print("GSPLAT_FAIL:", e)

# Look for scene cuda .so in this env
sp = sys.path
print("=== site-packages gsplat .so ===")
for p in sp:
    cand = os.path.join(p, "gsplat")
    if os.path.isdir(cand):
        for root, dirs, files in os.walk(cand):
            for fn in files:
                if fn.endswith(".so"):
                    fp = os.path.join(root, fn)
                    print("SO:", fp, os.path.getsize(fp))
