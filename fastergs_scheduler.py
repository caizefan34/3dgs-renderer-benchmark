#!/usr/bin/env python3
"""
Faster-GS training scheduler for 26 jobs across GPUs 0-6.
Each job runs fastergs_train.py with the appropriate config.
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from collections import deque

NERFICG_ROOT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg"
CONFIG_DIR = "/mnt/storage_pool/liaoyuanjun/strong_baselines/fastergs_configs"
RESULTS_BASE = "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/faster-gs"
TRAIN_SCRIPT = "/home/liaoyuanjun/3dgs-renderer-benchmark/fastergs_train.py"

SCENES = [
    ("mipnerf360", "bicycle"), ("mipnerf360", "flowers"), ("mipnerf360", "garden"),
    ("mipnerf360", "stump"), ("mipnerf360", "treehill"), ("mipnerf360", "room"),
    ("mipnerf360", "counter"), ("mipnerf360", "kitchen"), ("mipnerf360", "bonsai"),
    ("tanksandtemples", "truck"), ("tanksandtemples", "train"),
    ("deepblending", "drjohnson"), ("deepblending", "playroom"),
]

GPUS = [0, 1, 2, 3, 4, 5, 6]  # All 7 GPUs (evals done), skip GPU 7 (Candidate C)
MAX_GPUS = len(GPUS)

def get_output_dir(dataset, scene, method):
    return os.path.join(RESULTS_BASE, dataset, scene, method)

def is_done(dataset, scene, method):
    return os.path.exists(os.path.join(get_output_dir(dataset, scene, method), "training_summary.json"))

def launch_job(dataset, scene, method, gpu):
    """Launch a single training job."""
    output_dir = get_output_dir(dataset, scene, method)
    os.makedirs(output_dir, exist_ok=True)
    
    env = os.environ.copy()
    # Use nerficg conda env's python
    env["CUDA_HOME"] = "/home/liaoyuanjun/miniforge3"
    env["TORCH_HOME"] = "/home/liaoyuanjun/.cache/torch"
    torch_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                             "lib", "python3.11", "site-packages", "torch", "lib")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    
    log_file = os.path.join(output_dir, "scheduler.log")
    
    cmd = [
        sys.executable, "-u", TRAIN_SCRIPT,
        "--scene", scene,
        "--method", method,
        "--gpu", str(gpu),
        "--output_dir", output_dir,
    ]
    
    print(f"  [GPU {gpu}] faster-gs: {scene}/{method}", flush=True)
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                cwd="/home/liaoyuanjun/3dgs-renderer-benchmark")
    
    return proc

def main():
    # Build job queue
    jobs = deque()
    for dataset, scene in SCENES:
        for method in ["native", "c42"]:
            if not is_done(dataset, scene, method):
                jobs.append((dataset, scene, method))
    
    total = len(jobs)
    print(f"\n{'='*60}")
    print(f"  FASTER-GS TRAINING SCHEDULER")
    print(f"  {total} jobs to run on {MAX_GPUS} GPUs (0-{MAX_GPUS-1})")
    print(f"  GPU 7 reserved (Candidate C) — DO NOT TOUCH")
    print(f"{'='*60}\n", flush=True)
    
    running = {}  # gpu -> (proc, dataset, scene, method, start_time)
    done_count = 0
    failed_count = 0
    gpu_queue = deque(GPUS)
    
    while jobs or running:
        # Launch jobs on available GPUs
        while jobs and gpu_queue:
            gpu = gpu_queue.popleft()
            dataset, scene, method = jobs.popleft()
            proc = launch_job(dataset, scene, method, gpu)
            running[gpu] = (proc, dataset, scene, method, time.time())
        
        # Check for completed jobs
        if running:
            time.sleep(5)
            for gpu in list(running.keys()):
                proc, dataset, scene, method, start_time = running[gpu]
                ret = proc.poll()
                if ret is not None:
                    elapsed = (time.time() - start_time) / 60
                    if ret == 0:
                        print(f"  [GPU {gpu}] DONE: {scene}/{method} in {elapsed:.1f} min", flush=True)
                        done_count += 1
                    else:
                        print(f"  [GPU {gpu}] FAILED: {scene}/{method} exit={ret} in {elapsed:.1f} min", flush=True)
                        failed_count += 1
                    del running[gpu]
                    gpu_queue.append(gpu)
                    
                    remaining = len(jobs) + len(running)
                    print(f"  [status] running={len(running)}, remaining={remaining}, done={done_count}, failed={failed_count}", flush=True)
    
    print(f"\n{'='*60}")
    print(f"  FASTER-GS TRAINING COMPLETE")
    print(f"  Completed: {done_count}/{total}")
    print(f"  Failed: {failed_count}")
    print(f"{'='*60}\n", flush=True)
    
    # Save summary
    summary = {
        "baseline": "faster-gs",
        "total_jobs": total,
        "completed": done_count,
        "failed": failed_count,
    }
    with open(os.path.join(RESULTS_BASE, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
