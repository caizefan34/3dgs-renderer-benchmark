#!/usr/bin/env python3
"""
Scheduler for remaining C42 13-scene training jobs.
Launches jobs on free GPUs (0-6), skipping already-completed or running jobs.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
RESULTS_BASE = os.path.join(REPO, "results/c42_13scene")
GPU_LIST = [0, 1, 2, 3, 4, 5, 6]

# All jobs needed (10 scenes × 2 methods = 20)
ALL_JOBS = [
    ("mipnerf360", "flowers", "reference", 1.0),
    ("mipnerf360", "flowers", "c42", 0.5),
    ("mipnerf360", "stump", "reference", 1.0),
    ("mipnerf360", "stump", "c42", 0.5),
    ("mipnerf360", "treehill", "reference", 1.0),
    ("mipnerf360", "treehill", "c42", 0.5),
    ("mipnerf360", "counter", "reference", 1.0),
    ("mipnerf360", "counter", "c42", 0.5),
    ("mipnerf360", "kitchen", "reference", 1.0),
    ("mipnerf360", "kitchen", "c42", 0.5),
    ("mipnerf360", "bonsai", "reference", 1.0),
    ("mipnerf360", "bonsai", "c42", 0.5),
    ("tanksandtemples", "truck", "reference", 1.0),
    ("tanksandtemples", "truck", "c42", 0.5),
    ("tanksandtemples", "train", "reference", 1.0),
    ("tanksandtemples", "train", "c42", 0.5),
    ("deepblending", "drjohnson", "reference", 1.0),
    ("deepblending", "drjohnson", "c42", 0.5),
    ("deepblending", "playroom", "reference", 1.0),
    ("deepblending", "playroom", "c42", 0.5),
]

def is_job_done(dataset, scene, method):
    """Check if job already has training_results.json (completed)."""
    result_path = os.path.join(RESULTS_BASE, dataset, scene, method, "training_results.json")
    return os.path.exists(result_path)

def is_job_running(scene):
    """Check if a training process for this scene is currently running."""
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    return f"--scene {scene}" in result.stdout and "c42_13scene_train" in result.stdout

def get_busy_gpus():
    """Get set of GPU IDs that have compute processes."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    # Map GPU UUIDs to indices
    uuid_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    uuid_to_idx = {}
    for line in uuid_result.stdout.strip().split("\n"):
        parts = line.strip().split(", ")
        if len(parts) == 2:
            uuid_to_idx[parts[1]] = int(parts[0])
    
    busy = set()
    for line in result.stdout.strip().split("\n"):
        uuid = line.strip()
        if uuid in uuid_to_idx:
            idx = uuid_to_idx[uuid]
            if idx in GPU_LIST:  # Only care about our GPUs
                busy.add(idx)
    return busy

def launch_job(gpu_id, job):
    dataset, scene, method, scale = job
    output_dir = os.path.join(RESULTS_BASE, dataset, scene, method)
    log_file = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python3", "-u", os.path.join(REPO, "c42_13scene_train.py"),
        "--scene", scene, "--scale", str(scale),
        "--gpu", str(gpu_id), "--output_dir", output_dir,
        "--method", method,
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    
    print(f"  [GPU {gpu_id}] Launching: {scene}/{method} (scale={scale})", flush=True)
    
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                           cwd=REPO, env=env)
    return (job, proc, time.time(), log_file, log_fh)

# Filter out completed and running jobs
remaining = []
for job in ALL_JOBS:
    dataset, scene, method, scale = job
    if is_job_done(dataset, scene, method):
        print(f"  SKIP (done): {scene}/{method}", flush=True)
    elif is_job_running(scene):
        print(f"  SKIP (running): {scene}", flush=True)
    else:
        remaining.append(job)

print(f"\nRemaining jobs: {len(remaining)}/{len(ALL_JOBS)}", flush=True)
print(f"Available GPUs: {GPU_LIST}", flush=True)

# Track running jobs
running = {}  # gpu_id -> (job, proc, start_time, log_file, log_fh)
completed = []
failed = []

# Launch initial batch
for gpu_id in GPU_LIST:
    if not remaining:
        break
    busy = get_busy_gpus()
    if gpu_id not in busy:
        job = remaining.pop(0)
        running[gpu_id] = launch_job(gpu_id, job)

# Main scheduling loop
print(f"\nStarting scheduling loop...", flush=True)
iteration = 0

while remaining or running:
    iteration += 1
    
    # Check for completed jobs
    for gpu_id in list(running.keys()):
        job, proc, start_time, log_file, log_fh = running[gpu_id]
        ret = proc.poll()
        if ret is not None:
            log_fh.close()
            elapsed = time.time() - start_time
            if ret == 0:
                print(f"  [GPU {gpu_id}] COMPLETED: {job[1]}/{job[2]} in {elapsed/60:.1f} min", flush=True)
                completed.append((job, elapsed))
            else:
                print(f"  [GPU {gpu_id}] FAILED (exit {ret}): {job[1]}/{job[2]} in {elapsed/60:.1f} min", flush=True)
                failed.append((job, elapsed, ret))
            del running[gpu_id]
    
    # Launch new jobs on free GPUs
    busy = get_busy_gpus()
    for gpu_id in GPU_LIST:
        if gpu_id not in running and gpu_id not in busy and remaining:
            job = remaining.pop(0)
            running[gpu_id] = launch_job(gpu_id, job)
    
    if running:
        if iteration % 10 == 0:  # Print status every ~5 min
            print(f"  [status] running={len(running)}, remaining={len(remaining)}, "
                  f"completed={len(completed)}, failed={len(failed)}", flush=True)
        time.sleep(30)
    elif remaining:
        # No running jobs but still have remaining - wait for GPUs
        print(f"  [waiting] {len(remaining)} jobs remaining, waiting for GPUs...", flush=True)
        time.sleep(30)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"  TRAINING COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Completed: {len(completed)}/{len(ALL_JOBS)}", flush=True)
print(f"  Failed: {len(failed)}", flush=True)

for job, elapsed in completed:
    print(f"    OK: {job[1]}/{job[2]}: {elapsed/60:.1f} min", flush=True)
for job, elapsed, ret in failed:
    print(f"    FAIL: {job[1]}/{job[2]}: exit={ret}", flush=True)

# Save summary
summary = {
    "total_jobs": len(ALL_JOBS),
    "completed": len(completed),
    "failed": len(failed),
    "completed_jobs": [{"scene": j[1], "method": j[2], "elapsed_min": e/60} for j, e in completed],
    "failed_jobs": [{"scene": j[1], "method": j[2], "exit_code": r} for j, e, r in failed],
}
with open(os.path.join(RESULTS_BASE, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {RESULTS_BASE}/training_summary.json", flush=True)
