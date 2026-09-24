#!/usr/bin/env python3
"""
Launch missing training jobs on free GPUs.
Handles: counter/c42, kitchen ref/c42, bonsai ref/c42, playroom/c42
"""
import os, sys, time, json, subprocess

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
RESULTS_BASE = os.path.join(REPO, "results/c42_13scene")

MISSING_JOBS = [
    ("mipnerf360", "counter", "c42", 0.5),
    ("mipnerf360", "kitchen", "reference", 1.0),
    ("mipnerf360", "kitchen", "c42", 0.5),
    ("mipnerf360", "bonsai", "reference", 1.0),
    ("mipnerf360", "bonsai", "c42", 0.5),
    ("deepblending", "playroom", "c42", 0.5),
]

def is_done(dataset, scene, method):
    return os.path.exists(os.path.join(RESULTS_BASE, dataset, scene, method, "training_results.json"))

def is_running(scene, method):
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    return f"--scene {scene}" in result.stdout and f"--method {method}" in result.stdout and "c42_13scene_train" in result.stdout

def get_busy_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True)
    uuid_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True)
    uuid_to_idx = {}
    for line in uuid_result.stdout.strip().split("\n"):
        parts = line.strip().split(", ")
        if len(parts) == 2:
            uuid_to_idx[parts[1]] = int(parts[0])
    busy = set()
    for line in result.stdout.strip().split("\n"):
        uuid = line.strip()
        if uuid in uuid_to_idx and uuid_to_idx[uuid] < 7:
            busy.add(uuid_to_idx[uuid])
    return busy

def launch(gpu_id, job):
    dataset, scene, method, scale = job
    output_dir = os.path.join(RESULTS_BASE, dataset, scene, method)
    log_file = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["python3", "-u", os.path.join(REPO, "c42_13scene_train.py"),
           "--scene", scene, "--scale", str(scale),
           "--gpu", str(gpu_id), "--output_dir", output_dir, "--method", method]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"  [GPU {gpu_id}] Launching: {scene}/{method} (scale={scale})", flush=True)
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, cwd=REPO, env=env)
    return (job, proc, time.time(), log_fh)

# Filter truly missing jobs
to_launch = []
for job in MISSING_JOBS:
    dataset, scene, method, scale = job
    if is_done(dataset, scene, method):
        print(f"  SKIP (done): {scene}/{method}", flush=True)
    elif is_running(scene, method):
        print(f"  SKIP (running): {scene}/{method}", flush=True)
    else:
        to_launch.append(job)

print(f"\nJobs to launch: {len(to_launch)}", flush=True)

running = {}
completed = []
failed = []

while to_launch or running:
    # Check completed
    for gpu_id in list(running.keys()):
        job, proc, start_time, log_fh = running[gpu_id]
        ret = proc.poll()
        if ret is not None:
            log_fh.close()
            elapsed = time.time() - start_time
            if ret == 0:
                print(f"  [GPU {gpu_id}] DONE: {job[1]}/{job[2]} in {elapsed/60:.1f} min", flush=True)
                completed.append((job, elapsed))
            else:
                print(f"  [GPU {gpu_id}] FAIL: {job[1]}/{job[2]} exit={ret}", flush=True)
                failed.append((job, elapsed, ret))
            del running[gpu_id]
    
    # Launch on free GPUs
    busy = get_busy_gpus()
    for gpu_id in range(7):
        if gpu_id not in running and gpu_id not in busy and to_launch:
            job = to_launch.pop(0)
            running[gpu_id] = launch(gpu_id, job)
    
    if running:
        time.sleep(30)

print(f"\n=== DONE ===", flush=True)
print(f"Completed: {len(completed)}, Failed: {len(failed)}", flush=True)
