#!/usr/bin/env python3
"""
GPU scheduler for 13-scene benchmark training.
Launches training jobs across GPUs 0-6 (GPU 7 reserved for Candidate C).
Refills completed GPUs immediately.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from collections import OrderedDict

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
RESULTS_BASE = os.path.join(REPO, "results/c42_13scene")
GPU_LIST = [0, 1, 2, 3, 4, 5, 6]  # GPU 7 reserved for Candidate C

# 10 new scenes to train (3 existing scenes reuse checkpoints)
NEW_SCENES = [
    ("mipnerf360", "flowers"),
    ("mipnerf360", "stump"),
    ("mipnerf360", "treehill"),
    ("mipnerf360", "counter"),
    ("mipnerf360", "kitchen"),
    ("mipnerf360", "bonsai"),
    ("tanksandtemples", "truck"),
    ("tanksandtemples", "train"),
    ("deepblending", "drjohnson"),
    ("deepblending", "playroom"),
]

# Build job list: each job = (dataset, scene, method, scale, gpu)
JOBS = []
for dataset, scene in NEW_SCENES:
    JOBS.append((dataset, scene, "reference", 1.0))
    JOBS.append((dataset, scene, "c42", 0.5))

print(f"Total training jobs: {len(JOBS)}")
print(f"Available GPUs: {GPU_LIST}")
print(f"Estimated batches: {len(JOBS) / len(GPU_LIST):.1f}")

# Track running jobs: {gpu_id: (job, process, start_time, log_file)}
running = {}
completed = []
failed = []
job_queue = list(JOBS)

def launch_job(gpu_id, job):
    dataset, scene, method, scale = job
    output_dir = os.path.join(RESULTS_BASE, dataset, scene, method)
    log_file = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python3", os.path.join(REPO, "c42_13scene_train.py"),
        "--scene", scene,
        "--scale", str(scale),
        "--gpu", str(gpu_id),
        "--output_dir", output_dir,
        "--method", method,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"  [GPU {gpu_id}] Launching: {scene}/{method} (scale={scale})")
    
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                           cwd=REPO, env=env)
    
    return (job, proc, time.time(), log_file, log_fh)

def check_completed(gpu_id):
    job, proc, start_time, log_file, log_fh = running[gpu_id]
    ret = proc.poll()
    if ret is not None:
        log_fh.close()
        elapsed = time.time() - start_time
        if ret == 0:
            print(f"  [GPU {gpu_id}] COMPLETED: {job[1]}/{job[2]} in {elapsed/60:.1f} min")
            completed.append((job, elapsed))
        else:
            print(f"  [GPU {gpu_id}] FAILED (exit {ret}): {job[1]}/{job[2]} in {elapsed/60:.1f} min")
            failed.append((job, elapsed, ret))
        del running[gpu_id]
        return True
    return False

# Main scheduling loop
print(f"\nStarting training scheduler...")
batch_num = 0

while job_queue or running:
    # Launch jobs on free GPUs
    free_gpus = [g for g in GPU_LIST if g not in running]
    
    while free_gpus and job_queue:
        gpu_id = free_gpus.pop(0)
        job = job_queue.pop(0)
        running[gpu_id] = launch_job(gpu_id, job)
    
    # Check for completed jobs
    for gpu_id in list(running.keys()):
        check_completed(gpu_id)
    
    if running:
        time.sleep(30)  # Check every 30 seconds

# Summary
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Completed: {len(completed)}/{len(JOBS)}")
print(f"  Failed: {len(failed)}")

if completed:
    print(f"\n  Completed jobs:")
    for job, elapsed in completed:
        print(f"    {job[1]}/{job[2]}: {elapsed/60:.1f} min")

if failed:
    print(f"\n  Failed jobs:")
    for job, elapsed, ret in failed:
        print(f"    {job[1]}/{job[2]}: exit={ret}, {elapsed/60:.1f} min")

# Save summary
summary = {
    "total_jobs": len(JOBS),
    "completed": len(completed),
    "failed": len(failed),
    "completed_jobs": [{"scene": j[1], "method": j[2], "scale": j[3], "elapsed_min": e/60} 
                       for j, e in completed],
    "failed_jobs": [{"scene": j[1], "method": j[2], "scale": j[3], "exit_code": r, "elapsed_min": e/60}
                    for j, e, r in failed],
}
with open(os.path.join(RESULTS_BASE, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
