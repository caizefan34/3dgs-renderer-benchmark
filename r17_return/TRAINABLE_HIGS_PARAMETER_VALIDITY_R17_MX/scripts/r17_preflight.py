#!/usr/bin/env python3
"""R17 preflight: verify identities before any measurement.

Checks (no modification of anything):
  - frozen project tree + candidate/baseline csrc.so sha256 vs the R3/R8/R15 records
  - checkpoint exists + sha256 + N
  - bicycle cameras.json resolvable
  - R15 capture / R16 outputs presence (continuity references)
  - CUDA device + free memory snapshot
Output: raw/R17_PREFLIGHT.json (never blocks execution; violations are recorded)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, EXPECTED, OUT, PROJECT, R15_CAPTURE, R16_DIR,
                    add_project_to_syspath, camera_depth_stats, ensure_out_dirs,
                    gpu_mem_snapshot, load_checkpoint, resolve_cameras_json,
                    sha256_file, write_json)


def main():
    ensure_out_dirs()
    out = {"deliverable": "raw/R17_PREFLIGHT.json", "project": str(PROJECT)}
    checks = {}
    higs_bin = PROJECT / "gsplat/experimental/render/kernels/csrc.so"
    std_bin = PROJECT / "gsplat/csrc.so"
    checks["higs_csrc"] = {
        "path": str(higs_bin), "exists": higs_bin.is_file(),
        "sha256": sha256_file(higs_bin),
        "expected": EXPECTED["higs_csrc_sha256"],
        "match": sha256_file(higs_bin) == EXPECTED["higs_csrc_sha256"] if higs_bin.is_file() else None,
    }
    checks["std_csrc"] = {
        "path": str(std_bin), "exists": std_bin.is_file(),
        "sha256": sha256_file(std_bin),
        "expected": EXPECTED["std_csrc_sha256"],
        "match": sha256_file(std_bin) == EXPECTED["std_csrc_sha256"] if std_bin.is_file() else None,
    }
    ckpt, ck_info = load_checkpoint(CKPT, require=False)
    checks["checkpoint"] = {
        **ck_info,
        "expected_sha256": EXPECTED["ckpt_sha256"],
        "sha_match": ck_info.get("sha256") == EXPECTED["ckpt_sha256"],
        "n_match": ck_info.get("n_gaussians") == EXPECTED["n_gaussians"],
    }
    cams = resolve_cameras_json()
    checks["cameras_json"] = {"resolved": cams, "sha256": sha256_file(cams) if cams else None}
    checks["r15_capture"] = {"path": str(R15_CAPTURE), "exists": R15_CAPTURE.is_file(),
                             "sha256": sha256_file(R15_CAPTURE)}
    checks["r16_dir"] = {"path": str(R16_DIR), "exists": R16_DIR.is_dir(),
                         "csv": (R16_DIR / "R16_REPLAY_COUNTS.csv").is_file()}
    checks["cuda_preflight"] = gpu_mem_snapshot("preflight")
    try:
        import torch
        checks["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if ckpt is not None:
            import torch as _t
            m = ckpt["xyz"].float()
            if m.shape[0] > 100_000:
                idx = _t.randperm(m.shape[0])[:100_000]
                m = m[idx]
            checks["scene_vs_identity_camera"] = camera_depth_stats(
                m, _t.eye(4, dtype=_t.float32))
    except Exception as e:  # noqa: BLE001
        checks["torch"] = {"error": repr(e)}
    out["checks"] = checks

    def violated(k, c):
        if not isinstance(c, dict):
            return False
        if c.get("match") is False:
            return True
        if k == "checkpoint":
            return bool(c.get("exists")) and c.get("sha_match") is False
        return False

    out["blocking_violations"] = [k for k, c in checks.items() if violated(k, c)]
    write_json(OUT / "raw" / "R17_PREFLIGHT.json", out)
    print(json.dumps(out["blocking_violations"], indent=2), flush=True)
    print("preflight done (violations are recorded, not auto-blocking)", flush=True)


if __name__ == "__main__":
    add_project_to_syspath()
    main()
