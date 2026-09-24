#!/usr/bin/env python3
"""P2-1A-R2 remote env probe: core gsplat .so SHA, flat-op availability, fixtures."""
import hashlib
import json
import os
import torch
import sys

out = {"torch": torch.__version__, "cuda": torch.version.cuda,
       "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "device_count": torch.cuda.device_count()}

# gsplat core ABI
try:
    import gsplat
    from gsplat.csrc import _C
    p = _C.__file__
    out["gsplat_version"] = getattr(gsplat, "__version__", None)
    out["core_so"] = p
    out["core_so_sha"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    out["ops"] = {
        "intersect_tile": hasattr(torch.ops.gsplat, "intersect_tile"),
        "rasterize_to_pixels_3dgs": hasattr(torch.ops.gsplat, "rasterize_to_pixels_3dgs"),
    }
    import gsplat.rasterization as _r
    out["has_isect_offset_encode"] = hasattr(_r, "isect_offset_encode")
except Exception as e:
    out["gsplat_err"] = f"{type(e).__name__}: {e}"

# Fixture data dirs (mipnerf360 / nerfstudio style)
import glob
candidates = [
    "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/datasets",
    "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_data",
    "/mnt/storage_pool/liaoyuanjun/data",
    "/home/liaoyuanjun/3dgs-renderer-benchmark/data",
    "/home/liaoyuanjun/3dgs-renderer-benchmark/datasets",
]
found = {}
for base in candidates:
    if os.path.isdir(base):
        out.setdefault("candidate_data_roots", []).append(base)
        for scene in ("room", "bicycle", "garden"):
            for variant in (os.path.join(base, scene), os.path.join(base, scene, "camera_0"), os.path.join(base, scene, "cam0")):
                if os.path.isdir(variant):
                    found.setdefault(scene, []).append(variant)
out["fixture_candidates"] = found
print(json.dumps(out, sort_keys=True, indent=2))