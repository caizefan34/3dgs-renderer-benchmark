#!/usr/bin/env python3
"""Dataset adapter: map every P8/P9 method onto the canonical 13-scene layout.

Preparation artifact ONLY - it never launches training. It resolves data paths,
emits method-specific invocation arguments, and verifies that the canonical
files exist on the selected host.

Usage:
  python dataset_adapter.py --list-scenes
  python dataset_adapter.py --method fastgs --scene garden --host remote --emit-args
  python dataset_adapter.py --method speedy_splat --host local --verify
  python dataset_adapter.py --selftest

Exit codes: 0 ok / 1 missing files / 2 usage or consistency error.
"""

import argparse
import json
import os
import posixpath
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MIPNERF360 = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen",
              "room", "stump", "treehill"]
TANKSTEMPLES = ["truck", "train"]
DEEPBLENDING = ["drjohnson", "playroom"]
SCENES = {s: "mipnerf360" for s in MIPNERF360}
SCENES.update({s: "tanksandtemples" for s in TANKSTEMPLES})
SCENES.update({s: "deepblending" for s in DEEPBLENDING})

LOCAL_REPO = Path("C:/Users/36570/3dgs-renderer-benchmark")
LOCAL_DATASETS = LOCAL_REPO / "data" / "datasets"
LOCAL_OFFICIAL = LOCAL_REPO / "data" / "official" / "mipnerf360"
REMOTE_CANONICAL = "/mnt/storage_pool/liaoyuanjun/benchmark_data"

MATCHED_ARMS = ["b0", "gsplat_main", "taming_rasterizer", "b1a", "a0", "c0"]
P9_SYSTEMS = ["speedy_splat", "fastgs", "faster_gs"]

# fastgs per-scene overrides, transcribed verbatim from the frozen upstream
# train_base.sh @ 44e02a5 (public_baselines/fastgs/train_base.sh).
FASTGS_ARGS = {
    "bicycle":   {"grad_abs_thresh": "0.0012"},
    "flowers":   {"grad_abs_thresh": "0.0015", "dense": "0.005"},
    "garden":    {"grad_abs_thresh": "0.0008", "highfeature_lr": "0.02", "loss_thresh": "0.06"},
    "stump":     {"grad_abs_thresh": "0.0015", "dense": "0.004"},
    "treehill":  {"grad_abs_thresh": "0.002", "dense": "0.01"},
    "room":      {"grad_abs_thresh": "0.0008", "highfeature_lr": "0.02"},
    "counter":   {"grad_abs_thresh": "0.0008", "highfeature_lr": "0.02"},
    "kitchen":   {"grad_abs_thresh": "0.0006", "highfeature_lr": "0.02"},
    "bonsai":    {"grad_abs_thresh": "0.0006", "highfeature_lr": "0.02"},
    "truck":     {"grad_abs_thresh": "0.0009", "highfeature_lr": "0.04", "mult": "0.7"},
    "train":     {"grad_abs_thresh": "0.0015", "highfeature_lr": "0.042", "dense": "0.01", "mult": "0.7"},
    "playroom":  {"highfeature_lr": "0.0015", "dense": "0.003", "mult": "0.7"},
    "drjohnson": {"grad_abs_thresh": "0.0012", "highfeature_lr": "0.0025", "dense": "0.013", "mult": "0.7"},
}
FASTGS_COMMON = ["-i", "images", "--eval", "--densification_interval", "500",
                 "--optimizer_type", "default", "--test_iterations", "30000",
                 "--save_iterations", "30000", "--iterations", "30000"]


def scene_root(scene, host):
    fam = SCENES[scene]
    if host == "local":
        return LOCAL_DATASETS / fam / scene
    # keep remote paths POSIX even when the adapter runs on Windows
    return posixpath.join(REMOTE_CANONICAL, fam, scene)


def verify_scene(scene, host, need_official_cameras):
    """Return (ok, list of missing/notes)."""
    root = scene_root(scene, host)
    join = (lambda a, b: a / b) if host == "local" else posixpath.join
    isfile = (lambda p: Path(p).is_file()) if host == "local" else (lambda p: False)
    isdir = (lambda p: Path(p).is_dir()) if host == "local" else (lambda p: False)
    missing = []
    if host == "local" and not root.is_dir():
        return False, [f"scene dir missing: {root}"]
    for rel in ("sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"):
        if not isfile(join(root, rel)):
            missing.append(f"{rel} missing (or host=remote: run this check on mx)")
    img_dir = join(root, "images")
    if not isdir(img_dir):
        missing.append("images/ missing or empty (or host=remote: run this check on mx)")
    if need_official_cameras:
        # matched trainer (GTDataset) additionally reads official cameras.json + ply
        if host == "local":
            cam_json = LOCAL_OFFICIAL / scene / "cameras.json"
            if not cam_json.is_file():
                missing.append(f"official cameras.json missing: {cam_json}")
        else:
            missing.append("NOTE: official-cameras (GTDataset) layout on mx must be "
                           "verified there; canonical benchmark_data has sparse/0 only")
    return (not missing), missing


def emit_args(method, scene, host, outdir=None):
    """Return the method-specific invocation spec as a dict."""
    data = str(scene_root(scene, host))
    outdir = outdir or f"<output_dir>/{scene}"
    if method in MATCHED_ARMS:
        return {
            "method": method,
            "scene": scene,
            "data_source": "unified matched trainer (GTDataset: official cameras.json + datasets images; resolution=1080p)",
            "data_path": data,
            "seed": 42,
            "camera_sequence": "frozen camera_sequence.npy (identical across matched arms)",
            "iterations": 30000,
            "trainer": "scripts/publication/publication_trainer.py arm extension (b0 precedent: make_publication_trainer.py)",
        }
    if method == "speedy_splat":
        cmd = ["python", "-u", "train.py", "-s", data, "-m", outdir, "--eval",
               "--iterations", "30000", "--test_iterations", "30000",
               "--save_iterations", "30000"]
        return {"method": method, "scene": scene, "data_path": data,
                "native_recipe": "train.sh verbatim (no -r override; -i images default; auto-1600px)",
                "seed": "upstream safe_state seed 0 (recorded, not controlled)",
                "cmd": cmd}
    if method == "fastgs":
        cmd = ["python", "-u", "train.py", "-s", data, "-m", outdir] + FASTGS_COMMON
        for k, v in FASTGS_ARGS[scene].items():
            cmd += [f"--{k}", v]
        render_cmd = ["python", "render.py", "-m", outdir, "--skip_train"]
        if "mult" in FASTGS_ARGS[scene]:
            render_cmd += ["--mult", FASTGS_ARGS[scene]["mult"]]
        return {"method": method, "scene": scene, "data_path": data,
                "native_recipe": "train_base.sh @ 44e02a5 (frozen upstream table)",
                "seed": "upstream safe_state seed 0 (recorded, not controlled)",
                "cmd": cmd, "render_cmd": render_cmd,
                "metrics_cmd": ["python", "metrics.py", "-m", outdir]}
    if method == "faster_gs":
        scale = "0.25" if SCENES[scene] == "mipnerf360" else "1.0"
        return {"method": method, "scene": scene, "data_path": data,
                "native_recipe": "NeRFICG config-driven; base = fastergs_garden.yaml @ 117bdba",
                "seed": "RANDOM_SEED 0 (config)",
                "yaml_overrides": {
                    "PATH": data,
                    "IMAGE_SCALE_FACTOR": scale,
                    "TEST_STEP": 8,
                    "PPISP.USE": False,
                    "USE_MCMC": False,
                    "ITERATIONS": 30000,
                }}
    raise KeyError(f"unknown method: {method}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=MATCHED_ARMS + P9_SYSTEMS)
    ap.add_argument("--scene", choices=sorted(SCENES))
    ap.add_argument("--host", choices=["local", "remote"], default="local")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--emit-args", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list-scenes", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        problems = []
        for m in MATCHED_ARMS + P9_SYSTEMS:
            for s in SCENES:
                spec = emit_args(m, s, "remote")
                if spec.get("scene") != s:
                    problems.append(f"{m}/{s}: scene mismatch")
        for s, extra in FASTGS_ARGS.items():
            if s not in SCENES:
                problems.append(f"fastgs table references unknown scene {s}")
        if set(FASTGS_ARGS) != set(SCENES):
            problems.append("fastgs table does not cover exactly the 13 scenes")
        print(json.dumps({"selftest": "PASS" if not problems else "FAIL",
                          "problems": problems}, indent=2))
        return 2 if problems else 0

    if args.list_scenes:
        print(json.dumps({f: [s for s in SCENES if SCENES[s] == f]
                          for f in ("mipnerf360", "tanksandtemples", "deepblending")}, indent=2))
        return 0

    if not args.method:
        ap.error("--method is required unless --list-scenes/--selftest")

    if args.emit_args:
        if not args.scene:
            ap.error("--scene is required with --emit-args")
        print(json.dumps(emit_args(args.method, args.scene, args.host, args.outdir), indent=2))

    if args.verify:
        scenes = [args.scene] if args.scene else sorted(SCENES)
        need_cam = args.method in MATCHED_ARMS
        all_ok, report = True, {}
        for s in scenes:
            ok, missing = verify_scene(s, args.host, need_cam)
            report[s] = {"ok": ok, "missing": missing}
            all_ok &= ok
        print(json.dumps({"host": args.host, "method": args.method,
                          "all_ok": all_ok, "scenes": report}, indent=2))
        if not all_ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
