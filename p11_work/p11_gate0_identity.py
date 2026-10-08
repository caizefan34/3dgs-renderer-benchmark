#!/usr/bin/env python3
"""P11 Gate 0: source identity + strong baseline record.

No algorithm change. Records repo SHA, trainer SHA, gsplat .so SHA,
fused-ssim source/commit + .so SHA, torch/CUDA/driver/GPU UUID, exact
checkpoint identities.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_SOURCE_IDENTITY.json")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    ident = C.bootstrap_gmain()
    ns = C.import_recipe()

    # fused-ssim identity
    import fused_ssim
    import fused_ssim_cuda
    fso_path = fused_ssim_cuda.__file__
    fs_repo = "/root/fused-ssim"
    fs_commit = subprocess.run(
        ["git", "-C", fs_repo, "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    fs_subject = subprocess.run(
        ["git", "-C", fs_repo, "log", "-1", "--format=%ci %s"],
        capture_output=True, text=True).stdout.strip()
    fs_diff = subprocess.run(
        ["git", "-C", fs_repo, "status", "--porcelain"],
        capture_output=True, text=True).stdout.strip()

    # dataset identity per scene
    scenes = {}
    for scene in C.SCENES:
        seq = C.camera_sequence_for(scene)
        seq_path = os.path.join(C.FULL30K, f"gmain03_s42_{scene}",
                                "camera_sequence.npy")
        ckpt = os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "ckpt_30000.pt")
        camjson = os.path.join(C.REPO, "data", "official", "mipnerf360", scene, "cameras.json")
        scenes[scene] = {
            "ckpt_30000_path": ckpt,
            "ckpt_30000_sha256": C.sha256_file(ckpt) if os.path.exists(ckpt) else None,
            "ckpt_30000_size_bytes": os.path.getsize(ckpt) if os.path.exists(ckpt) else None,
            "camera_sequence_npy_sha256": C.sha256_file(seq_path)
                if os.path.exists(seq_path) else None,
            "cameras_json_sha256": C.sha256_file(camjson) if os.path.exists(camjson) else None,
            "campaign_command": open(os.path.join(
                C.FULL30K, f"gmain03_s42_{scene}", "command.txt")).read().strip()
                if os.path.exists(os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "command.txt")) else None,
            "campaign_launcher_meta": json.load(open(os.path.join(
                C.FULL30K, f"gmain03_s42_{scene}", "epic05_launcher_meta.json")))
                if os.path.exists(os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "epic05_launcher_meta.json")) else None,
        }

    # recipe source identities
    trainer_path = os.path.join(C.P8DIR, "publication_trainer_dsha.py")
    gaussian_model_path = os.path.join(C.REFERENCE_V1_DIR, "gaussian_model.py")
    config_path = os.path.join(C.REFERENCE_V1_DIR, "config.py")
    dataset_path = os.path.join(C.PHASE7_DIR, "dataset.py")

    # gsplat python package pin
    gmain_manifest = subprocess.run(
        ["bash", "-c",
         f"cd {C.GMAIN_TREE} && find gsplat -type f "
         f"\\( -name '*.py' -o -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \\) "
         f"| grep -v third_party | grep -v __pycache__ | sort | xargs sha256sum | sha256sum"],
        capture_output=True, text=True).stdout.strip()

    # fused-ssim window constant check (source-level semantic evidence)
    window_check = check_window()

    record = {
        "phase": "P11 Gate 0 - identity and strong baseline",
        "date_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                   capture_output=True, text=True).stdout.strip(),
        "upstream": {
            "repository": "nerfstudio-project/gsplat",
            "commit": "512d366b67073d77ca099ede742683c165dfc23b",
            "gsplat_version": "1.6.0",
            "tree": C.GMAIN_TREE,
            "gsplat_python_plus_cuda_source_manifest_sha256": gmain_manifest,
            "build_identity_file": "/root/epic05_p8/gmain_build_identity.json",
        },
        "gsplat_so": {
            "path": C.GMAIN_SO,
            "sha256": ident["so_sha256"],
            "matches_p8_gmain_build": ident["so_sha256"] ==
                json.load(open("/root/epic05_p8/gmain_build_identity.json"))["so_sha256"],
        },
        "trainer": {
            "path": trainer_path,
            "sha256": C.sha256_file(trainer_path),
            "arm": "gmain",
            "recipe": "DSH-A publication: eps2d=0.3, absgrad=True (all iters), "
                      "tile_size=16, packed=False, rasterize_mode=classic, "
                      "SepSSIM 11x11 sigma=1.5 same-pad, L=0.8*L1+0.2*(1-SSIM), "
                      "densify 500..15000@100, grow_grad2d=0.0008 absgrad pixel-space, "
                      "opacity reset 3000, SH progress 1000, Adam eps=1e-15, seed=42",
            "gaussian_model_sha256": C.sha256_file(gaussian_model_path),
            "config_sha256": C.sha256_file(config_path),
            "dataset_sha256": C.sha256_file(dataset_path),
        },
        "fused_ssim": {
            "repo": "https://github.com/rahul-goel/fused-ssim",
            "clone_path": fs_repo,
            "commit": fs_commit,
            "commit_subject": fs_subject,
            "working_tree_clean": fs_diff == "",
            "module_file": fused_ssim.__file__,
            "so_path": fso_path,
            "so_sha256": C.sha256_file(fso_path),
            "build_flags": ["-O3", "--use_fast_math", "--maxrregcount=32",
                            "CUDA_ARCHITECTURES=80"],
            "padding_mode_used": "same",
            "note": "existing public implementation used as strong baseline; "
                    "NOT a P11 contribution",
            "window_constant_check": window_check,
        },
        "environment": C.env_identity(),
        "gpu_used": C.assert_clean_gpu(args.gpu),
        "scenes": scenes,
        "frozen_states_policy": {
            "late_30k": "full30k/gmain03_s42_{scene}/ckpt_30000.pt (DSH-A campaign, "
                        "P8_STRICT_MATCHED_0P3, seed 42)",
            "pre15k": "P11 stategen: faithful replay of the same trainer/recipe to iter "
                      "14000 saving capture()+optimizer at 2000/5000/10000/14000 "
                      "(state generation only; NOT a 30K campaign; measurement-only)",
        },
        "no_algorithm_change": True,
    }
    C.dump_json(record, args.out)


def check_window():
    """Compare fused-ssim's hardcoded cGauss[11] with the trainer's torch kernel."""
    import re
    src = open("/root/fused-ssim/ssim.cu").read()
    m = re.search(r"cGauss\[11\]\s*=\s*\{(.*?)\};", src, re.S)
    fused_vals = [float(x) for x in m.group(1).replace("f", "").split(",") if x.strip()]
    coords = torch.arange(11, dtype=torch.float32) - 5
    k1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    k1d = k1d / k1d.sum()
    diff = max(abs(a - float(b)) for a, b in zip(fused_vals, k1d.tolist()))
    return {
        "fused_cgauss": fused_vals,
        "trainer_k1d": k1d.tolist(),
        "max_abs_diff": diff,
        "bitwise_equal": all(
            a == float(b) for a, b in zip(fused_vals, k1d.tolist())),
        "note": "fused-ssim hardcodes the 11-tap sigma=1.5 Gaussian; comparison "
                "against the trainer's torch.exp-derived kernel",
    }


if __name__ == "__main__":
    main()
