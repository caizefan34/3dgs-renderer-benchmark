#!/usr/bin/env python3
"""A0-P10 Stage 0 -- remote identity / hash verification of the frozen P12/C0 stack.

Run on EPIC-05 (A100) with the frozen gsplat env python:
    /root/miniforge3/envs/gsplat/bin/python a0p10_identity.py --out /root/A0_P10/out

Verifies, by sha256, every frozen source the P12/C0 baseline was certified against,
plus:
  - room/25K checkpoint sha256 (lead-provided authoritative value);
  - core gsplat_cuda.so sha256 (frozen C0 lineage);
  - c0 ext .so sha256 (P12 publication lineage);
  - BYTE-IDENTITY of the authoritative _moonshot kernel sources
    (HigsNativeBackward.cu, RasterizeToPixels3DGSDevice.cuh) vs the remote frozen
    C0 files (sha256 equality);
  - frozen camera index 19 img_name == DSCF4819.
Writes A0_SOURCE_IDENTITY.json and prints a PASS/FAIL summary.

This script does NOT render, backward, or train -- it only verifies provenance so
that any P10 structure data is traceable to the exact frozen P12/C0 baseline.
"""
import argparse
import json
import os
import sys
import platform
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import (  # noqa: E402
    REPO, C0_WT, C0_SO, CORE_SO, FUSED_SSIM_SO, FUSED_SSIM_MOD, P12_TRAINER,
    REFERENCE_V1_DIR, PHASE7_DIR, FROZEN_SHA256, CKPT_DIR, RECIPE, P10_SCENE, P10_CKPT,
    P10_CAM_INDEX, P10_CAM_IMAGE, REMOTE_KERNEL_CU, REMOTE_KERNEL_CUH,
    LOCAL_KERNEL_CU, LOCAL_KERNEL_CUH, sha256_file,
)

EXPECTED = [
    # (label, path, expected_sha256 or None to record-only)
    ("core_so gsplat_cuda (frozen C0 lineage)", CORE_SO, FROZEN_SHA256["core_so"]),
    ("c0_ext_so (P12 publication lineage)", C0_SO, FROZEN_SHA256["c0_ext_so"]),
    ("fused_ssim_cuda.so (P11-locked loss backend)", FUSED_SSIM_SO, FROZEN_SHA256["fused_ssim_so"]),
    ("fused_ssim/__init__.py", FUSED_SSIM_MOD, None),
    ("P12 trainer publication_trainer_p12.py", P12_TRAINER, FROZEN_SHA256["p12_trainer"]),
    ("gaussian_model.py", os.path.join(REFERENCE_V1_DIR, "gaussian_model.py"), FROZEN_SHA256["gaussian_model"]),
    ("config.py", os.path.join(REFERENCE_V1_DIR, "config.py"), FROZEN_SHA256["config"]),
    ("colmap_reader.py", os.path.join(REFERENCE_V1_DIR, "colmap_reader.py"), FROZEN_SHA256["colmap_reader"]),
    ("phase7 dataset.py", os.path.join(PHASE7_DIR, "dataset.py"), FROZEN_SHA256["phase7_dataset"]),
    ("C0 renderer gaussian_inference.py (C0 worktree)",
     os.path.join(C0_WT, "gsplat/experimental/render/functional/gaussian_inference.py"),
     FROZEN_SHA256["higs_renderer_py"]),
]

# Authoritative frozen kernel sources: byte-identity vs the remote frozen C0 files.
KERNEL_BYTE_IDENTITY = [
    # (label, remote_path, local_reference_path)
    ("HigsNativeBackward.cu byte-identity", REMOTE_KERNEL_CU, LOCAL_KERNEL_CU),
    ("RasterizeToPixels3DGSDevice.cuh byte-identity", REMOTE_KERNEL_CUH, LOCAL_KERNEL_CUH),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    checks = []

    def add(label, path, expected):
        present = os.path.exists(path)
        if not present:
            checks.append({"label": label, "path": path, "present": False,
                           "sha256": None, "expected": expected, "ok": False,
                           "note": "FILE MISSING"})
            return
        sha = sha256_file(path)
        if expected:
            ok = (sha == expected)
            note = "MATCH" if ok else "HASH MISMATCH"
        else:
            ok = True
            note = "RECORDED (no frozen sha asserted)"
        checks.append({"label": label, "path": path, "present": True,
                       "sha256": sha, "expected": expected, "ok": ok, "note": note})

    def add_byte_identity(label, remote_path, local_ref):
        rp = os.path.normpath(remote_path)
        lp = os.path.normpath(local_ref)
        r_present = os.path.exists(rp)
        l_present = os.path.exists(lp)
        # If the REMOTE file is missing, the byte-identity cannot be evaluated; this is
        # recorded as a non-fatal note (the path layout can differ between hosts), so it
        # does not hard-block the run. A present-but-mismatching remote file IS a hard
        # failure (that would mean the frozen source was altered).
        if not r_present:
            checks.append({"label": label, "remote_path": rp, "local_ref": lp,
                           "present": False,
                           "remote_sha256": None,
                           "local_sha256": sha256_file(lp) if l_present else None,
                           "ok": True,
                           "note": "REMOTE PATH MISSING (byte-identity not evaluated; non-fatal)"})
            return
        rsha = sha256_file(rp)
        lsha = sha256_file(lp)
        ok = (rsha == lsha)
        checks.append({"label": label, "remote_path": rp, "local_ref": lp,
                       "present": True, "remote_sha256": rsha, "local_sha256": lsha,
                       "ok": ok, "note": "BYTE-IDENTICAL" if ok else "BYTE MISMATCH"})

    for label, path, expected in EXPECTED:
        add(label, path, expected)
    for label, rp, lp in KERNEL_BYTE_IDENTITY:
        add_byte_identity(label, rp, lp)

    # Checkpoint room/25K (authoritative sha from lead review)
    ckpt_path = os.path.join(CKPT_DIR.format(scene=P10_SCENE), f"ckpt_{P10_CKPT}.pt")
    add(f"checkpoint {P10_SCENE}/{P10_CKPT}", ckpt_path, FROZEN_SHA256["ckpt_room_25000"])

    # Camera index 19 img_name assertion (DSCF4819). Best-effort from the dataset
    # cameras.json (no GPU needed). If the dataset is not loadable in this env,
    # the check is recorded as PENDING (the a0p10_run.py driver re-asserts it on
    # the live GTDataset before any capture).
    cam_img = None
    try:
        import json as _json
        cam_json = os.path.join(REPO, "data", "official", "mipnerf360", P10_SCENE, "cameras.json")
        if os.path.exists(cam_json):
            cams = _json.load(open(cam_json))
            cam_img = cams[P10_CAM_INDEX].get("img_name")
    except Exception as e:
        cam_img = None
    cam_ok = (cam_img == P10_CAM_IMAGE) if cam_img is not None else False
    checks.append({"label": f"frozen camera index {P10_CAM_INDEX} img_name == {P10_CAM_IMAGE}",
                   "camera_index": P10_CAM_INDEX, "img_name": cam_img,
                   "present": cam_img is not None, "ok": cam_ok,
                   "note": ("MATCH" if cam_ok else
                            ("PENDING (dataset not loadable in identity env; "
                             "re-asserted on live GTDataset by a0p10_run.py)" if cam_img is None else
                             f"EXPECTED {P10_CAM_IMAGE} GOT {cam_img}"))})

    # Environment / recipe (recorded; C0 env is set by bootstrap_c0 at runtime)
    env_block = {
        "hostname": platform.node(),
        "python": sys.executable,
        "torch": None, "cuda_runtime": None, "gpu": None,
        "c0_env_expected": {
            "HIGS_PX_RUNTIME": "2",
            "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
            "HIGS_BWD_H8_MR": "1",
            "HIGS_BWD_ABSGRAD": "1",
            "HIGS_DISABLE_F9": None,  # C0 = F9 ON
        },
        "recipe": RECIPE,
        "frozen_constants": {
            "MAX_ALPHA": 0.99,
            "ALPHA_THRESHOLD": 1.0 / 255.0,
            "TRANSMITTANCE_THRESHOLD": 1e-4,
            "MIN_ONE_MINUS_ALPHA": 1e-6,
            "source": "_moonshot/Common.h == p18_p0/c0_source/Common.h (byte-identical)",
        },
    }
    try:
        import torch
        env_block["torch"] = torch.__version__
        env_block["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            env_block["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # torch import is optional for identity
        env_block["torch_note"] = f"torch import failed: {e}"

    all_ok = all(c["ok"] for c in checks)
    result = {
        "stage": "0_identity",
        "campaign": "A0-P10_PREFLIGHT",
        "ts_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "baseline": "frozen P12/C0",
        "scene": P10_SCENE, "ckpt": P10_CKPT,
        "identity_checks": checks,
        "environment": env_block,
        "all_present": all(c["present"] for c in checks),
        "all_hash_ok": all_ok,
        "pass": all(c["ok"] and c["present"] for c in checks),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"=== A0-P10 Stage 0 identity -> {a.out}")
    for c in checks:
        flag = "OK " if c["ok"] and c["present"] else ("MISS" if not c["present"] else "FAIL")
        sha = c.get("sha256") or c.get("remote_sha256") or (c.get("img_name") or "")
        print(f"  [{flag}] {c['label']}: {str(sha)[:20]}.. {c['note']}")
    print(f"  torch={env_block['torch']} cuda={env_block['cuda_runtime']} gpu={env_block['gpu']}")
    print(f"  PASS={result['pass']}")
    if not result["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
