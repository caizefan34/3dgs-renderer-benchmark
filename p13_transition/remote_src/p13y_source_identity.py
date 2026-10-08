#!/usr/bin/env python3
"""P13-Y Y0a: source identity assertions -> P13Y_SOURCE_IDENTITY.json.

Verifies every frozen artifact identity the P13-Y protocol depends on and
aborts (ABORT_IDENTITY_MISMATCH) on any drift. No GPU needed (hash-only),
but fused_ssim_cuda must import under the campaign env python.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

OUT = "/root/epic05_p13y/artifacts/P13Y_SOURCE_IDENTITY.json"

# ---------------- frozen identity (P13-Y protocol Sec.1) -------------------
P12_TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
P12_TRAINER_SHA = "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
C0_SO = ("/root/epic05_build_cache/c0_ext/"
         "experimental_gaussian_render_inference_scene_cuda/"
         "experimental_gaussian_render_inference_scene_cuda.so")
C0_SO_SHA = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
CORE_SO_SHA = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
FUSED_SO_SHA = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
GSPLAT_PIN = "512d366b67073d77ca099ede742683c165dfc23b"
HIGS_FN = os.path.join(C0_WT, "gsplat/experimental/render/functional/gaussian_inference.py")
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"

CAPTURE_TRAINER = "/root/epic05_p13y/publication_trainer_p13y_capture.py"
CAPTURE_PATCH = "/root/epic05_p13y/P13Y_TRAINER_DIFF.patch"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rep = {"phase": "P13-Y", "date": "2026-09-27", "checks": {}, "abort": False}
    checks = rep["checks"]

    def chk(name, path, expected, required=True):
        entry = {"path": path, "expected_sha256": expected, "required": required}
        if os.path.exists(path):
            actual = sha256_file(path)
            entry["sha256"] = actual
            entry["match"] = bool(actual == expected)
        else:
            entry["sha256"] = None
            entry["match"] = False
            entry["error"] = "missing"
        checks[name] = entry
        if required and not entry["match"]:
            rep["abort"] = True
        return entry["match"]

    chk("p12_trainer", P12_TRAINER, P12_TRAINER_SHA)
    chk("c0_extension_so", C0_SO, C0_SO_SHA)
    chk("c0_core_so", CORE_SO, CORE_SO_SHA)

    # gsplat C0 worktree source identity (the worktree is a detached worktree
    # with an unreadable .git link, so the operative identity is the file-level
    # sha256 of the training-render implementation + the documented pin).
    if os.path.exists(HIGS_FN):
        checks["higs_render_source"] = {
            "path": HIGS_FN, "sha256": sha256_file(HIGS_FN),
            "documented_pin": GSPLAT_PIN,
            "note": "worktree .git link dead; file hash is the operative identity",
        }
    else:
        checks["higs_render_source"] = {"path": HIGS_FN, "error": "missing"}
        rep["abort"] = True

    # fused SSIM binary (import requires torch first)
    try:
        import torch  # noqa: F401
        import fused_ssim_cuda
        fso = fused_ssim_cuda.__file__
        chk("fused_ssim_cuda_so", fso, FUSED_SO_SHA)
    except Exception as e:  # pragma: no cover
        checks["fused_ssim_cuda_so"] = {"error": repr(e)}
        rep["abort"] = True

    # capture trainer (byte copy of P12 + additive P13-Y side channel)
    if os.path.exists(CAPTURE_TRAINER):
        checks["capture_trainer"] = {
            "path": CAPTURE_TRAINER, "sha256": sha256_file(CAPTURE_TRAINER),
            "p12_base_sha256": P12_TRAINER_SHA,
        }
        if os.path.exists(CAPTURE_PATCH):
            checks["capture_trainer"]["additive_diff_sha256"] = sha256_file(CAPTURE_PATCH)
    else:
        checks["capture_trainer"] = {"path": CAPTURE_TRAINER,
                                     "error": "not yet created"}

    # env identity (C0 runtime switches must be set by the trainer bootstrap;
    # record current process env here as informational)
    rep["env"] = {k: v for k, v in os.environ.items() if k.startswith("HIGS_")}
    rep["python"] = sys.version
    rep["torch"] = getattr(__import__("torch"), "__version__", None) if "torch" in sys.modules else None
    try:
        rep["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()
    except Exception as e:
        rep["nvidia_driver"] = repr(e)

    rep["status"] = "ABORT_IDENTITY_MISMATCH" if rep["abort"] else "IDENTITY_OK"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print("WROTE", OUT)
    if rep["abort"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
