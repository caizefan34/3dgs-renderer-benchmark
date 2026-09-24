#!/usr/bin/env python3
"""
Faster-GS training script for NeRFICG framework.
Trains one scene/method using a pre-generated YAML config.
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

NERFICG_ROOT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg"
CONFIG_DIR = "/mnt/storage_pool/liaoyuanjun/strong_baselines/fastergs_configs"
RESULTS_BASE = "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/faster-gs"
NERFICG_OUTPUT = os.path.join(NERFICG_ROOT, "output", "FasterGS")

def train_fastergs(scene, method, gpu, output_dir):
    """Train Faster-GS native or C42."""
    config_path = os.path.join(CONFIG_DIR, f"{scene}_{method}.yaml")

    if not os.path.exists(config_path):
        print(f"  ERROR: config not found: {config_path}", flush=True)
        return 1

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCH_HOME"] = "/home/liaoyuanjun/.cache/torch"
    # Set LD_LIBRARY_PATH for nerficg env's torch lib
    torch_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                             "lib", "python3.11", "site-packages", "torch", "lib")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    # NeRFICG stores output in OUTPUT_DIR/METHOD_TYPE/MODEL_NAME_TIMESTAMP
    # We override MODEL_NAME to be unique and find the output after training
    model_name = f"{scene}_{method}"

    cmd = [
        sys.executable, "-u", os.path.join(NERFICG_ROOT, "scripts", "train.py"),
        "-c", config_path,
        "TRAINING.MODEL_NAME=" + model_name,
        "GLOBAL.GPU_INDICES=[0]",  # GPU 0 after CUDA_VISIBLE_DEVICES filtering
    ]

    log_file = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  [GPU {gpu}] Training Faster-GS {scene} {method}", flush=True)
    print(f"  Config: {config_path}", flush=True)

    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                            cwd=NERFICG_ROOT)

    # Find the output directory (model_name_TIMESTAMP) and symlink it to our output_dir
    if proc.returncode == 0:
        import glob
        pattern = os.path.join(NERFICG_OUTPUT, f"{model_name}_*")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if candidates:
            nerficg_out = candidates[0]
            # Copy key results to our output dir
            for item in os.listdir(nerficg_out):
                src = os.path.join(nerficg_out, item)
                dst = os.path.join(output_dir, item)
                if os.path.islink(dst) or os.path.exists(dst):
                    continue
                if os.path.isdir(src):
                    os.symlink(src, dst)
                else:
                    import shutil
                    shutil.copy2(src, dst)
            print(f"  Linked results from {nerficg_out} to {output_dir}", flush=True)

    return proc.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", required=True, choices=["native", "c42"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    t_start = time.time()
    ret = train_fastergs(args.scene, args.method, args.gpu, args.output_dir)
    elapsed = time.time() - t_start

    summary = {
        "baseline": "faster-gs",
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
