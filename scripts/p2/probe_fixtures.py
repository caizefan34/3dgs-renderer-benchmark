#!/usr/bin/env python3
"""Verify exact A100 fixture checkpoint + cameras paths and resolution for room/bicycle/garden."""
import json, os, math
CKPT_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"
CAM_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
SCENES = ["room", "bicycle", "garden"]
out = {}
for s in SCENES:
    ckpt = f"{CKPT_BASE}/a100_30k_{s}_t16_16/a100_30k_{s}_t16_16_latest.pt"
    cams = f"{CAM_BASE}/{s}/cameras.json"
    e = out[s] = {"ckpt_exists": os.path.isfile(ckpt), "cams_exists": os.path.isfile(cams),
                  "ckpt_size": os.path.getsize(ckpt) if os.path.isfile(ckpt) else None}
    if os.path.isfile(cams):
        with open(cams) as f:
            c = json.load(f)
        e["n_cams"] = len(c)
        cam0 = c[0]
        e["cam0"] = {k: cam0.get(k) for k in ("width","height","fx","fy","rotation","position") if k in cam0}
        # max_long_side=2048 downscale factor
        W, H = cam0.get("width"), cam0.get("height")
        long_side = max(W, H)
        sf = max(1.0, long_side / 2048)
        e["max_long_side_scaling"] = sf
        e["scaled_res"] = [int(round(W/sf)), int(round(H/sf))]
print(json.dumps(out, sort_keys=True, indent=2))