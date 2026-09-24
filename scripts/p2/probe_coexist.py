#!/usr/bin/env python3
"""Verify: load core ABI so for torch.ops.gsplat ops + R2 so for experimental op, coexist; locate fixtures."""
import json, os, hashlib, importlib.util, sys
import torch

out = {}

CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2 = "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"

def load_so(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

errs = {}
# Load core ABI first
try:
    core = load_so(CORE, "core_abi")
    out["core_loaded"] = True
    out["core_ops"] = {
        "intersect_tile": hasattr(torch.ops.gsplat, "intersect_tile"),
        "rasterize_to_pixels_3dgs": hasattr(torch.ops.gsplat, "rasterize_to_pixels_3dgs"),
    }
except Exception as e:
    out["core_loaded"] = False
    errs["core"] = f"{type(e).__name__}: {e}"

# Load R2 so
try:
    r2 = load_so(R2, "r2_mod")
    out["r2_loaded"] = True
    out["r2_has_entry"] = hasattr(r2, "higs_native_hierarchy_from_projected")
    out["r2_has_producer"] = hasattr(r2, "higs_gatherless_projected_producer")
    out["r2_schema"] = str(torch.ops.experimental.higs_native_hierarchy_from_projected.default._schema)
except Exception as e:
    out["r2_loaded"] = False
    errs["r2"] = f"{type(e).__name__}: {e}"

out["errors"] = errs

# Locate fixtures
import glob
roots = ["/mnt/storage_pool/liaoyuanjun/data",
         "/home/liaoyuanjun/3dgs-renderer-benchmark/data",
         "/mnt/nas", "/mnt/storage_pool/liaoyuanjun"]
hits = {}
for root in roots:
    if not os.path.isdir(root):
        continue
    for scene in ("room", "bicycle", "garden", "cam0", "mipnerf360/room"):
        g = glob.glob(os.path.join(root, "**", scene), recursive=True)
        for d in g[:10]:
            hits.setdefault(scene, []).append(d)
out["fixture_hits"] = hits
print(json.dumps(out, sort_keys=True, indent=2))