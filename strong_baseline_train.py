#!/usr/bin/env python3
"""
Unified training script for strong baselines (Speedy-Splat and FastGS).
Handles both native (c42_scale=1.0) and C42 (c42_scale=0.5) variants.
Records timing, peak VRAM, Gaussian count, and final evaluation metrics.

Usage:
  python strong_baseline_train.py --baseline speedy-splat --scene bicycle --method native --gpu 0 --output_dir /path/to/output
  python strong_baseline_train.py --baseline speedy-splat --scene bicycle --method c42 --gpu 1 --output_dir /path/to/output
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

def train_speedysplat(scene, method, gpu, output_dir, data_root):
    """Train Speedy-Splat native or C42."""
    repo = "/mnt/storage_pool/liaoyuanjun/strong_baselines/speedy-splat"
    data_path = os.path.join(data_root, "speedy-splat/datasets/mipnerf360", scene)
    
    # Map scene to dataset
    tt_scenes = ["truck", "train"]
    db_scenes = ["drjohnson", "playroom"]
    if scene in tt_scenes:
        data_path = os.path.join(data_root, "speedy-splat/datasets/tanksandtemples", scene)
    elif scene in db_scenes:
        data_path = os.path.join(data_root, "speedy-splat/datasets/deepblending", scene)
    
    c42_scale = "0.5" if method == "c42" else "1.0"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Fix library path - use conda env's torch lib
    torch_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                             "lib", "python3.10", "site-packages", "torch", "lib")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = repo
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = [
        sys.executable, "-u", os.path.join(repo, "train.py"),
        "-s", data_path,
        "-m", output_dir,
        "--eval",
        "--c42_scale", c42_scale,
        "--iterations", "30000",
        "--test_iterations", "30000",
        "--save_iterations", "30000",
    ]
    
    log_file = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  [GPU {gpu}] Training Speedy-Splat {scene} {method} (c42_scale={c42_scale})", flush=True)
    
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=repo)
    
    return proc.returncode

def train_fastgs(scene, method, gpu, output_dir, data_root):
    """Train FastGS native or C42."""
    repo = "/mnt/storage_pool/liaoyuanjun/strong_baselines/fastgs"
    data_path = os.path.join(data_root, "fastgs/datasets/mipnerf360", scene)
    
    tt_scenes = ["truck", "train"]
    db_scenes = ["drjohnson", "playroom"]
    if scene in tt_scenes:
        data_path = os.path.join(data_root, "fastgs/datasets/tanksandtemples", scene)
    elif scene in db_scenes:
        data_path = os.path.join(data_root, "fastgs/datasets/deepblending", scene)
    
    c42_scale = "0.5" if method == "c42" else "1.0"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                             "lib", "python3.10", "site-packages", "torch", "lib")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = repo
    env["PYTHONUNBUFFERED"] = "1"
    
    # FastGS-specific args from train_base.sh
    cmd = [
        sys.executable, "-u", os.path.join(repo, "train.py"),
        "-s", data_path,
        "-i", "images",
        "-m", output_dir,
        "--eval",
        "--c42_scale", c42_scale,
        "--iterations", "30000",
        "--test_iterations", "30000",
        "--save_iterations", "30000",
        "--densification_interval", "500",
        "--optimizer_type", "default",
        "--grad_abs_thresh", "0.0012",
    ]
    
    # Add mult for T&T and DB scenes (compact box multiplier)
    if scene in tt_scenes or scene in db_scenes:
        cmd.extend(["--mult", "0.7"])
    
    log_file = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  [GPU {gpu}] Training FastGS {scene} {method} (c42_scale={c42_scale})", flush=True)
    
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=repo)
    
    return proc.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["speedy-splat", "fastgs"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", required=True, choices=["native", "c42"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="/mnt/storage_pool/liaoyuanjun/strong_baselines")
    args = parser.parse_args()
    
    t_start = time.time()
    
    if args.baseline == "speedy-splat":
        ret = train_speedysplat(args.scene, args.method, args.gpu, args.output_dir, args.data_root)
    else:
        ret = train_fastgs(args.scene, args.method, args.gpu, args.output_dir, args.data_root)
    
    elapsed = time.time() - t_start
    
    # Save training summary
    summary = {
        "baseline": args.baseline,
        "scene": args.scene,
        "method": args.method,
        "gpu": args.gpu,
        "exit_code": ret,
        "wall_time_s": elapsed,
        "wall_time_min": elapsed / 60,
    }
    
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    if ret == 0:
        print(f"  COMPLETED: {args.scene}/{args.method} in {elapsed/60:.1f} min", flush=True)
    else:
        print(f"  FAILED: {args.scene}/{args.method} exit={ret}", flush=True)
