#!/usr/bin/env python3
"""R18 preflight — verify environment, binaries, checkpoint, cameras, GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r18_common import (EXPECTED, H540, W540, add_project_to_syspath, gpu_mem_snapshot,
                        load_checkpoint, read_common_constants, resolve_cameras_json,
                        sha256_file, write_json)

add_project_to_syspath()
PY = sys.executable

checks = {}
checks["higs_csrc"] = {
    "path": str(PROJECT / "gsplat/experimental/render/kernels/csrc.so"),
    "exists": (PROJECT / "gsplat/experimental/render/kernels/csrc.so").is_file(),
    "sha256": sha256_file(PROJECT / "gsplat/experimental/render/kernels/csrc.so"),
    "expected": EXPECTED["higs_csrc_sha256"],
}
checks["higs_csrc"]["match"] = checks["higs_csrc"]["sha256"] == EXPECTED["higs_csrc_sha256"]
checks["std_csrc"] = {
    "path": str(PROJECT / "gsplat/csrc.so"),
    "exists": (PROJECT / "gsplat/csrc.so").is_file(),
    "sha256": sha256_file(PROJECT / "gsplat/csrc.so"),
    "expected": EXPECTED["std_csrc_sha256"],
}
checks["std_csrc"]["match"] = checks["std_csrc"]["sha256"] == EXPECTED["std_csrc_sha256"]

ckpt, ck = load_checkpoint(require=False)
checks["checkpoint"] = ck
checks["checkpoint"]["n_gaussians"] = ck.get("n_gaussians")
checks["checkpoint"]["match"] = ck.get("sha256") == EXPECTED["ckpt_sha256"]

cj = resolve_cameras_json()
checks["cameras_json"] = {"resolved": cj, "sha256": sha256_file(cj) if cj else None}
try:
    cams, _, _ = __import__("r18_common").load_real_cameras(W540, H540, device="cpu")
    checks["cameras_json"]["n_cams_540p"] = len(cams)
    checks["cameras_json"]["cam0_img_name"] = cams[0]["img_name"] if cams else None
except Exception as e:  # noqa: BLE001
    checks["cameras_json"]["error"] = repr(e)

checks["common_constants"] = read_common_constants()
try:
    import torch
    checks["cuda"] = {"available": bool(torch.cuda.is_available()),
                       "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    checks["gpu_mem"] = gpu_mem_snapshot("preflight")
    import gsplat  # noqa: F401
    checks["gsplat_import"] = "OK"
    from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen  # noqa: F401
    checks["higs_frozen_import"] = "OK"
except Exception as e:  # noqa: BLE001
    checks["cuda"] = {"error": repr(e)}

blocking = []
if not checks["higs_csrc"]["match"]:
    blocking.append("higs_csrc_hash_mismatch")
if not checks["std_csrc"]["match"]:
    blocking.append("std_csrc_hash_mismatch")
if not checks["checkpoint"].get("match"):
    blocking.append("checkpoint_hash_mismatch")
if not checks["cameras_json"].get("resolved"):
    blocking.append("cameras_json_missing")

out = {"deliverable": "R18_PREFLIGHT", "project": str(PROJECT),
       "checks": checks, "blocking_violations": blocking,
       "python": PY}
write_json(OUT / "R18_PREFLIGHT.json", out)
print(json.dumps(out, indent=2, default=str))
print("PREFLIGHT done (violations recorded, not auto-blocking)" if not blocking
      else f"PREFLIGHT blocking: {blocking}")
sys.exit(0)
