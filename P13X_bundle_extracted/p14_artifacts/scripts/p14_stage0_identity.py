"""P14 Stage 0: source identity verification + prior-art boundary doc.

ABORT_IDENTITY_MISMATCH on any prereg hash mismatch (P14 protocol §0/§2).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C

FUSED_SSIM_CANDIDATES = [
    os.path.join(C.C0_WT, "fused_ssim.so"),
    "/root/epic05_build_cache/fused_ssim/fused_ssim.so",
]
TRAINER_CANDIDATES = [
    "/root/epic05_p12/publication_trainer_p12.py",
    "/root/epic05_p13x/publication_trainer_p13x.py",
]


def main():
    os.makedirs(C.P14_ROOT, exist_ok=True)
    for d in ("logs", "artifacts", "raw", "P14_O1_UNITS", "P14_O2_UNITS"):
        os.makedirs(os.path.join(C.P14_ROOT, d), exist_ok=True)

    checks = {}
    abort = False

    def check(name, path, expected):
        nonlocal abort
        if not os.path.exists(path):
            checks[name] = {"path": path, "exists": False,
                            "expected": expected, "match": False}
            abort = True
            return
        h = C.sha256_file(path)
        ok = (h == expected)
        checks[name] = {"path": path, "exists": True, "sha256": h,
                        "expected": expected, "match": ok}
        if not ok:
            abort = True

    check("c0_ext_so", C.C0_SO, C.PREREG_HASHES["c0_ext_so"])
    check("core_so", C.CORE_SO, C.PREREG_HASHES["core_so"])
    # resolve fused_ssim_cuda via the gsplat env's own import path (the
    # trainer locks this exact module at runtime)
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import fused_ssim_cuda; print(fused_ssim_cuda.__file__)"],
            capture_output=True, text=True, timeout=120)
        fs_path = out.stdout.strip().splitlines()[-1]
    except Exception:
        fs_path = None
    if not fs_path or not os.path.exists(fs_path):
        fs_path = os.path.join(
            "/root/miniforge3/envs/gsplat/lib/python3.10/"
            "site-packages",
            "fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so")
    assert os.path.exists(fs_path), "fused_ssim_cuda.so not found"
    check("fused_ssim_so", fs_path, C.PREREG_HASHES["fused_ssim_so"])
    check("p12_trainer", TRAINER_CANDIDATES[0],
          C.PREREG_HASHES["p12_trainer"])

    # checkpoint provenance (P13-X verified 16K states)
    ck = {}
    for s in C.SCENES:
        p = C.CKPT[s]
        ck[s] = {"path": p, "exists": os.path.exists(p),
                 "sha256": C.sha256_file(p) if os.path.exists(p) else None,
                 "size_bytes": os.path.getsize(p)
                 if os.path.exists(p) else None}

    py = sys.version.split()[0]
    import torch
    # informational: the P13-X trainer copy carries diagnostic hooks
    # added AFTER the P13-X identity snapshot, so it is EXPECTED to
    # differ from the pristine P12 trainer; it is not a P14 dependency
    # (P14 consumes only the 16K checkpoints + frozen binaries).
    p13x_trainer_sha = (C.sha256_file(TRAINER_CANDIDATES[1])
                        if os.path.exists(TRAINER_CANDIDATES[1]) else None)
    ident = {
        "phase": "P14",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("Source identity snapshot before any P14 measurement; "
                    "P14 operates on the P13-X verified frozen C0 16K "
                    "states without modifying any frozen binary."),
        "frozen_binaries": checks,
        "p13x_trainer_copy_informational": {
            "path": TRAINER_CANDIDATES[1],
            "sha256": p13x_trainer_sha,
            "note": ("expected to differ from P12: P13-X diagnostic "
                     "hooks were added after the P13-X identity "
                     "snapshot; not a P14 dependency"),
        },
        "checkpoints_16k": ck,
        "prereg": {
            "seed": 42, "packed": False, "tile": 16, "eps2d": 0.3,
            "renderer": "classic C0 (rasterize_gaussian_higs_dynamic)",
            "resolution": "1080p", "background": "black",
            "HIGS env": {"HIGS_PX_RUNTIME": "2",
                         "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
                         "HIGS_BWD_H8_MR": "1", "HIGS_BWD_ABSGRAD": "1"},
        },
        "env": {"python": py, "torch": torch.__version__,
                "hostname": os.uname().nodename,
                "gpu_snapshot_at_stage0": C.gpu_snapshot()},
        "verdict_identity": None,
    }
    if abort:
        ident["verdict_identity"] = "ABORT_IDENTITY_MISMATCH"
        C.jwrite(os.path.join(C.P14_ROOT, "artifacts",
                              "P14_SOURCE_IDENTITY.json"), ident)
        print("ABORT_IDENTITY_MISMATCH", json.dumps(checks, indent=2))
        sys.exit(2)
    ident["verdict_identity"] = "IDENTITY_VERIFIED"
    C.jwrite(os.path.join(C.P14_ROOT, "artifacts",
                          "P14_SOURCE_IDENTITY.json"), ident)
    print("[p14] identity verified: all prereg hashes match")

    # ---- prior-art boundary (P14 protocol §2) -----------------------
    boundary = """# P14 Prior-Art Boundary (preregistered)

## Explicitly NOT claimed as novel (prior art)

- Gaussian grouping / partitioning G = G_A + G_F (Group Training and
  related work partition Gaussians into groups).
- Freezing + caching a subset of Gaussians to save computation.
  Group Training's cached group is REMOVED from the rendered image.
- Opacity-prioritized sampling of active Gaussians (Speedy-Splat style
  opacity-ordered work reduction is prior art).
- Visible-only parameter updates (visibility-masked update rules are
  standard in official 3DGS code paths).
- Sparse / selective optimizer updates: Sparse Adam, gsplat
  SelectiveAdam (Taming3DGS) are prior art, benchmarked as prior art.
- Asynchronous primitive update of cached components (async caches are
  prior art in rendering systems).

## P14's single tested distinction

Frozen G_F must still FULLY participate in the rendered image
I = R(G_A, G_F) - never R(G_A) - while optimization-induced stationarity
(frozen params + frozen Adam state within a micro-epoch) is exploited to
compress forward, backward, and optimizer work. The novelty status of
this exact compiled-frozen-context mechanism is UNPROVEN by P14; P14
only measures whether the oracle upper bound can pass the frozen gates.

## Status

NOVELTY_STATUS = UNPROVEN. No novelty claim is made by P14 regardless of
outcome. No speedup claim beyond the measured oracle numbers. The
masked-optimizer benchmark (SelectiveAdam / equivalent) is prior art and
is NOT a P14 contribution.
"""
    with open(os.path.join(C.P14_ROOT, "artifacts",
                           "P14_PRIOR_ART_BOUNDARY.md"), "w") as f:
        f.write(boundary)
    print("[p14] prior-art boundary written")


if __name__ == "__main__":
    main()
