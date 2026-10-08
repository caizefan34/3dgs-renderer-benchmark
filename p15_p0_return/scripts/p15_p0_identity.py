"""P15-P0 Step 2 - source identity (re-run of the P14 identity check).

ABORT_IDENTITY_MISMATCH on any fatal mismatch. Writes
artifacts/P15_SOURCE_IDENTITY.json into the P15-P0 tree. No measurement,
no GPU work, no training.

Frozen checks (preregistered before any P15-P0 export result):

  A. FATAL - the four P14 prereg binary hashes, identical set and
     semantics to p14_stage0_identity.py (the P14 scripts are NOT edited):
       c0_ext_so, core_so, fused_ssim_so, p12_trainer
  B. FATAL - the two P13-X 16K checkpoints: sha256 must match the values
     recorded in P14_SOURCE_IDENTITY.json (and P14's manifest).
  C. FATAL - certified reuse surface:
       scripts/p14_common.py and scripts/p14_selector.py must match
       MANIFEST_P14.json["scripts"] (P15 imports them in place).
  D. FATAL - P15 input evidence against MANIFEST_P14.json["artifacts"]:
       artifacts/P14_CAMERA_CLUSTERS.json
       artifacts/P14_SELECTOR.json
       raw/P14_ACTIVE_room.npz / raw/P14_ACTIVE_bicycle.npz
  E. FATAL - the six H=32 P14 O1 anchor units exist for every cell and
     every q in {0.10,0.20,0.25,0.30,0.50} (D1 + F).
  F. INFO  - auxiliary P14 artifacts (O1/O2 aggregates, verdict,
     contamination, unit counts) hashed and recorded.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jwrite(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[p15] wrote {path}", flush=True)


Q5 = (0.10, 0.20, 0.25, 0.30, 0.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p14-scripts", default="/root/epic05_p14/scripts")
    ap.add_argument("--p14-root", default="/root/epic05_p14")
    ap.add_argument("--p15-root", default="/root/epic05_p15/p0")
    args = ap.parse_args()

    sys.path.insert(0, args.p14_scripts)
    import p14_common as C
    import torch

    A = lambda *p: os.path.join(args.p14_root, "artifacts", *p)
    R = lambda *p: os.path.join(args.p14_root, *p)

    fatal = {}
    info = {}

    def check_fatal(name, path, expected, note=""):
        if not os.path.exists(path):
            fatal[name] = {"path": path, "exists": False,
                           "expected": expected, "match": False,
                           "note": note}
            return
        h = sha256_file(path)
        fatal[name] = {"path": path, "exists": True, "sha256": h,
                       "expected": expected, "match": (h == expected),
                       "note": note}

    # ---- A. prereg binaries (same set as P14 stage 0) ------------------
    check_fatal("c0_ext_so", C.C0_SO, C.PREREG_HASHES["c0_ext_so"])
    check_fatal("core_so", C.CORE_SO, C.PREREG_HASHES["core_so"])
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
            "/root/miniforge3/envs/gsplat/lib/python3.10/site-packages",
            "fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so")
    check_fatal("fused_ssim_so", fs_path, C.PREREG_HASHES["fused_ssim_so"])
    check_fatal("p12_trainer",
                "/root/epic05_p12/publication_trainer_p12.py",
                C.PREREG_HASHES["p12_trainer"])

    # ---- C/D. manifest chain ------------------------------------------
    man_path = A("MANIFEST_P14.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    info["MANIFEST_P14.json"] = {
        "path": man_path, "exists": os.path.exists(man_path),
        "sha256": sha256_file(man_path) if os.path.exists(man_path)
        else None}

    for base in ("p14_common.py", "p14_selector.py"):
        expected = manifest.get("scripts", {}).get(base)
        check_fatal(f"reuse_{base}", os.path.join(args.p14_scripts, base),
                    expected, note="certified reuse surface (read-only)")

    p15_inputs = {
        "P14_CAMERA_CLUSTERS.json": R("artifacts", "P14_CAMERA_CLUSTERS.json"),
        "P14_SELECTOR.json": R("artifacts", "P14_SELECTOR.json"),
        "P14_ACTIVE_room.npz": R("raw", "P14_ACTIVE_room.npz"),
        "P14_ACTIVE_bicycle.npz": R("raw", "P14_ACTIVE_bicycle.npz"),
    }
    for name, path in p15_inputs.items():
        prefix = "artifacts/" if name.endswith(".json") else "raw/"
        rec = manifest.get("artifacts", {}).get(prefix + name)
        expected = rec.get("sha256") if isinstance(rec, dict) else rec
        check_fatal(f"input_{name}", path, expected,
                    note="P15 input evidence")

    # ---- B. checkpoints ------------------------------------------------
    p14_ident_path = A("P14_SOURCE_IDENTITY.json")
    p14_ident = json.load(open(p14_ident_path)) \
        if os.path.exists(p14_ident_path) else {}
    ck = {}
    for s in C.SCENES:
        p = C.CKPT[s]
        rec = p14_ident.get("checkpoints_16k", {}).get(s, {})
        expected = rec.get("sha256")
        exists = os.path.exists(p)
        h = sha256_file(p) if exists else None
        ck[s] = {"path": p, "exists": exists, "sha256": h,
                 "expected": expected, "match": bool(expected and h == expected),
                 "size_bytes": os.path.getsize(p) if exists else None}
        if not ck[s]["match"]:
            fatal[f"ckpt_{s}"] = ck[s]

    # ---- E. anchor units ------------------------------------------------
    units_missing = []
    for s in C.SCENES:
        for ci in range(3):
            for qq in (10, 20, 25, 30, 50):
                for tag in (f"d1_h32", f"f_q{qq}_h32"):
                    p = R("P14_O1_UNITS", f"{s}_cell{ci}_{tag}.json")
                    if not os.path.exists(p):
                        units_missing.append(p)
    if units_missing:
        fatal["anchor_units_missing"] = {"missing": units_missing}
    info["anchor_units"] = {"required": 6 * 5 * 2,
                            "missing": len(units_missing)}

    # ---- F. auxiliary info ---------------------------------------------
    for name in ("P14_O1_AGG.json", "P14_O2_AGG.json", "P14_VERDICT.json",
                 "P14_CONTAMINATION.json", "P14_PRIOR_ART_BOUNDARY.md"):
        p = A(name)
        info[name] = {"exists": os.path.exists(p),
                      "sha256": sha256_file(p) if os.path.exists(p)
                      else None}
    n_units = 0
    udir = R("P14_O1_UNITS")
    if os.path.isdir(udir):
        n_units = len([f for f in os.listdir(udir) if f.endswith(".json")])
    info["P14_O1_UNITS_count"] = n_units

    bad = [k for k, v in fatal.items()
           if (not v.get("match", False))]
    abort = bool(bad)

    ident = {
        "phase": "P15-P0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("source identity snapshot before any P15-P0 export; "
                    "P15-P0 consumes the frozen P13-X 16K states and the "
                    "certified P14 scripts/binaries without modifying "
                    "anything"),
        "fatal_checks": fatal,
        "info_checks": info,
        "checkpoints_16k": ck,
        "prereg": {
            "seed": 42, "tile": 16, "eps2d": 0.3,
            "renderer": "classic C0 (rasterize_gaussian_higs_dynamic)",
            "resolution": "1080p", "background": "black",
            "HIGS env": {"HIGS_PX_RUNTIME": "2",
                         "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
                         "HIGS_BWD_H8_MR": "1",
                         "HIGS_BWD_ABSGRAD": "1"},
            "cost_definition": ("exact Gaussian-tile intersection count "
                                "(cull -> gatherless projected producer -> "
                                "intersect_tile), summed over the cell's 8 "
                                "cluster cameras at the 16K initial state"),
            "progress_proxy": "P14 selector energy S_i (float64)",
        },
        "env": {"python": sys.version.split()[0],
                "torch": torch.__version__,
                "hostname": os.uname().nodename,
                "gpu_snapshot": C.gpu_snapshot()},
        "verdict_identity": None,
    }
    if abort:
        ident["verdict_identity"] = "ABORT_IDENTITY_MISMATCH"
        jwrite(os.path.join(args.p15_root, "artifacts",
                            "P15_SOURCE_IDENTITY.json"), ident)
        print("ABORT_IDENTITY_MISMATCH")
        for k in bad:
            print(f"  {k}: {json.dumps(fatal[k])[:200]}")
        sys.exit(2)
    ident["verdict_identity"] = "IDENTITY_VERIFIED"
    jwrite(os.path.join(args.p15_root, "artifacts",
                        "P15_SOURCE_IDENTITY.json"), ident)
    print("[p15] identity verified: P14 prereg binaries, checkpoints, "
          "reuse surface, P15 input evidence and H=32 anchors all match")


if __name__ == "__main__":
    main()
