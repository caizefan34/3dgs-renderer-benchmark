#!/usr/bin/env python
# rf0a_make_meta.py - generate RF0A source-lock / environment / camera-split / config JSON
import json, os, sys, subprocess, hashlib, platform, re

REPO = "/root/rf0a_laneA/FastGS"
DATA = "/root/epic05_p16rb/common_data/mipnerf360"
SCENES = ["room", "garden", "bicycle"]
SEED = 42
ITER = 30000
OUTDIR = "/root/rf0a_laneA/final"
os.makedirs(OUTDIR, exist_ok=True)

def run(cmd, cwd=REPO):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True).stdout.strip()

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def git(*args):
    return run("git " + " ".join(args))

# ---------- SOURCE LOCK ----------
head = git("rev-parse HEAD")
remote = git("config --get remote.origin.url")
branch = git("rev-parse --abbrev-ref HEAD")
describe = git("describe --tags --always") or head
status = run("git status --porcelain")
submodules = run("git submodule status")
gitlinks = run("git ls-files -s submodules")
n_gitlinks = len([l for l in gitlinks.splitlines() if l.strip().startswith("160000")])
n_tracked_sub = len([l for l in run("git ls-files submodules").splitlines() if l.strip()])

def tree_hash(relpath):
    return run(f"git rev-parse HEAD:{relpath}")

built_ext = {}
for name, rel in [
    ("diff_gaussian_rasterization_fastgs", "submodules/diff-gaussian-rasterization_fastgs"),
    ("simple_knn", "submodules/simple-knn"),
    ("fused_ssim", "submodules/fused-ssim"),
]:
    so_paths = []
    for root, dirs, files in os.walk(os.path.join(REPO, rel)):
        for fn in files:
            if fn.endswith(".so"):
                so_paths.append(os.path.join(root, fn))
    # also site-packages installs
    entry = {"repo_path": rel, "repo_tree_sha256": None, "built_so_sha256": []}
    tp = os.path.join(REPO, rel)
    if os.path.isdir(tp):
        entry["repo_tree_sha256"] = run(f"git rev-parse HEAD:{rel}")
    for sp in so_paths:
        entry["built_so_sha256"].append({"path": sp, "sha256": sha256_file(sp)})
    built_ext[name] = entry

# site-packages .so (installed)
sp = sys.prefix + "/lib/python3.12/site-packages"
for name, moddir in [("diff_gaussian_rasterization_fastgs", "diff_gaussian_rasterization_fastgs"),
                     ("simple_knn", "simple_knn"), ("fused_ssim", "fused_ssim")]:
    d = os.path.join(sp, moddir)
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            for fn in files:
                if fn.endswith(".so"):
                    p = os.path.join(root, fn)
                    built_ext[name].setdefault("site_packages_so", []).append({"path": p, "sha256": sha256_file(p)})

source_lock = {
    "lane": "RF0A (ReproFast0 Lane A)",
    "remote_url": remote,
    "branch": branch,
    "head_commit_sha": head,
    "git_describe": describe,
    "working_tree_clean": (status == ""),
    "git_status_porcelain": status,
    "submodules_declared": submodules if submodules else "NONE (no .gitmodules; rasterizer/simple-knn/fused-ssim are vendored into the main tree)",
    "gitlink_submodules": n_gitlinks,
    "vendored_tracked_files_under_submodules": n_tracked_sub,
    "vendored_component_tree_shas": {
        "diff_gaussian_rasterization_fastgs": tree_hash("submodules/diff-gaussian-rasterization_fastgs"),
        "simple_knn": tree_hash("submodules/simple-knn"),
        "fused_ssim": tree_hash("submodules/fused-ssim"),
    },
    "built_extensions": built_ext,
    "notes": "Official fastgs/FastGS @ commit 44e02a5 (CVPR 2026). CUDA extensions are vendored (not git submodules); the single commit pins their source. _C.so built on EPIC-05 against torch 2.10.0+cu128 / nvcc 12.8.",
}

# ---------- ENVIRONMENT ----------
import torch
nvcc = run("nvcc --version | tail -1")
gcc = run("gcc --version | head -1")
pip_pkgs = run("python -m pip list --format=freeze")
env = {
    "host": platform.node(),
    "os": run("uname -a"),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    "gpu": run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1"),
    "gpu_count": torch.cuda.device_count(),
    "nvcc": nvcc,
    "gcc": gcc,
    "cmake": run("cmake --version | head -1"),
    "ninja": run("ninja --version"),
    "pip_packages": pip_pkgs,
}

# ---------- CAMERA SPLIT ----------
sys.path.insert(0, REPO)
from scene.dataset_readers import readColmapSceneInfo
split = {"rule": "original camera index i (COLMAP image order) with i % 8 == 0 => HOLDOUT (test); all others => train. Holdout cameras are NEVER used for training (train.py samples only getTrainCameras()).", "llffhold": 8, "eval_flag": True, "scenes": {}}
for sc in SCENES:
    info = readColmapSceneInfo(os.path.join(DATA, sc), "images", eval=True, llffhold=8)
    tr = info.train_cameras; te = info.test_cameras
    split["scenes"][sc] = {
        "source_path": os.path.join(DATA, sc),
        "total_cameras": len(tr) + len(te),
        "train_count": len(tr),
        "test_count": len(te),
        "cameras_extent_radius": round(float(info.nerf_normalization["radius"]), 6),
        "test_image_names": [c.image_name for c in te],
        "test_indices_mod8_eq_0": True,
    }

# ---------- CONFIG ----------
# Official FastGS per-scene Mip-NeRF360 settings from train_base.sh (verbatim knobs).
fastgs_scene_args = {
    "room":   {"highfeature_lr": "0.02", "grad_abs_thresh": "0.0008"},
    "garden": {"highfeature_lr": "0.02", "loss_thresh": "0.06", "grad_abs_thresh": "0.0008"},
    "bicycle":{"grad_abs_thresh": "0.0008"},
}
common_fastgs = ["--eval", "--densification_interval", "500", "--optimizer_type", "default", "--test_iterations", "30000"]
vanilla_args = ["--eval", "--iterations", "30000", "--densification_interval", "100",
                "--densify_grad_threshold", "0.0002", "--percent_dense", "0.01",
                "--densify_from_iter", "500", "--densify_until_iter", "15000",
                "--opacity_reset_interval", "3000", "--position_lr_init", "0.00016",
                "--position_lr_final", "0.0000016", "--position_lr_max_steps", "30000",
                "--position_lr_delay_mult", "0.01", "--feature_lr", "0.0025",
                "--opacity_lr", "0.05", "--scaling_lr", "0.005", "--rotation_lr", "0.001",
                "--lambda_dssim", "0.2", "--shfeature_lr", "0.0025"]
config = {
    "seed": SEED,
    "iterations": ITER,
    "split_rule": "i % 8 == 0 => HOLDOUT (true holdout, never trained on)",
    "primary_metric": "corrected true-holdout PSNR/SSIM/LPIPS",
    "arms": {
        "A_fastgs": {
            "description": "Official FastGS (train_base.sh per-scene Mip-NeRF360 settings), verbatim train.py loop, mult=0.5, optimizer_type=default.",
            "mult": 0.5, "eval_mult": 0.5,
            "per_scene_extra_args": fastgs_scene_args,
            "common_args": common_fastgs,
        },
        "B_vanilla": {
            "description": "Faithful original INRIA 3DGS comparator on the SAME backend (same diff_gaussian_rasterization_fastgs rasterizer, same Scene/GaussianModel). Differs from FastGS only in: original mean-gradient clone/split densification; per-iteration dense Adam; mult=1.0 (full compact box); no final_prune_fastgs. Canonical 3DGS Mip-NeRF360 hyperparameters.",
            "mult": 1.0, "eval_mult": 1.0,
            "args": vanilla_args,
            "comparator_justification": "Official fastgs/FastGS exposes NO vanilla baseline in-repo (grep for vanilla render/densify_and_prune is empty; optimizer_type only selects default vs sparse_adam, both FastGS densification). Per task spec, the faithful vanilla/comparator is the original INRIA 3DGS the repo is forked from (LICENSE_ORIGINAL.md), run on the identical rasterizer backend so wall-time differences reflect the algorithm, not the backend.",
        },
    },
    "measurement": {
        "training_wall_seconds": "end-to-end python wall clock of the training loop (excludes ~40s camera load)",
        "internal_training_time_seconds": "sum of CUDA-event iter+optim timing (matches train.py 'Training time')",
        "peak_vram_gb": "torch.cuda.max_memory_allocated() over the run",
        "final_gaussian_count": "gaussians._xyz.shape[0] at iter 30000",
        "holdout_metrics": "PSNR/SSIM/LPIPS over true-holdout (test) cameras, rendered with each arm's own mult",
        "gaussian_trajectory": "n_gaussians logged every 1000 iters",
    },
}

for fn, obj in [("RF0A_SOURCE_LOCK.json", source_lock), ("RF0A_ENVIRONMENT.json", env),
                ("RF0A_CAMERA_SPLIT.json", split), ("RF0A_CONFIG.json", config)]:
    with open(os.path.join(OUTDIR, fn), "w") as f:
        json.dump(obj, f, indent=2)
    print("WROTE", os.path.join(OUTDIR, fn))
print("META_DONE")
