#!/usr/bin/env python3
"""
Scheduler for strong baseline 13-scene training.
Launches jobs across GPUs 0-6, refills as GPUs free up.
Supports both Speedy-Splat and FastGS baselines.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
STRONG_BASE = "/mnt/storage_pool/liaoyuanjun/strong_baselines"
RESULTS_BASE = "/mnt/storage_pool/liaoyuanjun/strong_baseline_results"
GPU_LIST = [0, 1, 2, 3, 4, 5, 6]

ALL_SCENES = [
    ("mipnerf360", "bicycle"),
    ("mipnerf360", "flowers"),
    ("mipnerf360", "garden"),
    ("mipnerf360", "stump"),
    ("mipnerf360", "treehill"),
    ("mipnerf360", "room"),
    ("mipnerf360", "counter"),
    ("mipnerf360", "kitchen"),
    ("mipnerf360", "bonsai"),
    ("tanksandtemples", "truck"),
    ("tanksandtemples", "train"),
    ("deepblending", "drjohnson"),
    ("deepblending", "playroom"),
]

def is_job_done(baseline, dataset, scene, method):
    result_path = os.path.join(RESULTS_BASE, baseline, dataset, scene, method, "training_summary.json")
    if not os.path.exists(result_path):
        return False
    with open(result_path) as f:
        s = json.load(f)
    return s.get("exit_code", 1) == 0

def is_running(baseline, dataset, scene, method):
    """Check if a specific scene/method training is already running by matching the model path in the command line."""
    model_path = os.path.join(RESULTS_BASE, baseline, dataset, scene, method)
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "train.py" in line and model_path in line:
            return True
    return False

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

def launch_job(gpu_id, baseline, dataset, scene, method):
    output_dir = os.path.join(RESULTS_BASE, baseline, dataset, scene, method)
    log_file = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)

    repo = os.path.join(STRONG_BASE, "speedy-splat" if baseline == "speedy-splat" else "fastgs")
    data_path = os.path.join(STRONG_BASE, baseline, "datasets", dataset, scene)

    c42_scale = "0.5" if method == "c42" else "1.0"

    # Build command — assign unique GUI port using a global counter to avoid conflicts
    gui_port = 7000 + hash((scene, method, gpu_id)) % 1000
    cmd = [sys.executable, "-u", os.path.join(repo, "train.py"),
           "-s", data_path, "-m", output_dir, "--eval",
           "--c42_scale", c42_scale,
           "--iterations", "30000",
           "--test_iterations", "30000",
           "--save_iterations", "30000",
           "--port", str(gui_port)]

    if baseline == "fastgs":
        cmd.extend(["-i", "images", "--densification_interval", "500",
                    "--optimizer_type", "default", "--grad_abs_thresh", "0.0012"])
        if dataset in ("tanksandtemples", "deepblending"):
            cmd.extend(["--mult", "0.7"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Use conda env's torch lib (derived from sys.executable)
    torch_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                             "lib", "python3.10", "site-packages", "torch", "lib")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = repo
    env["PYTHONUNBUFFERED"] = "1"

    print(f"  [GPU {gpu_id}] {baseline}: {scene}/{method} (c42_scale={c42_scale})", flush=True)

    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, cwd=repo)
    return (proc, time.time(), log_fh)

if __name__ == "__main__":
    BASELINE = sys.argv[1] if len(sys.argv) > 1 else "speedy-splat"

    # Build job list
    all_jobs = []
    for dataset, scene in ALL_SCENES:
        all_jobs.append((dataset, scene, "native"))
        all_jobs.append((dataset, scene, "c42"))

    # Filter done/running
    remaining = []
    for dataset, scene, method in all_jobs:
        if is_job_done(BASELINE, dataset, scene, method):
            print(f"  SKIP (done): {scene}/{method}", flush=True)
        elif is_running(BASELINE, dataset, scene, method):
            print(f"  SKIP (running): {scene}/{method}", flush=True)
        else:
            remaining.append((dataset, scene, method))

    print(f"\n{BASELINE}: {len(remaining)}/{len(all_jobs)} jobs to launch", flush=True)

    running = {}
    completed = []
    failed = []

    # Launch initial batch
    busy = get_busy_gpus()
    for gpu_id in GPU_LIST:
        if not remaining:
            break
        if gpu_id not in busy:
            job = remaining.pop(0)
            running[gpu_id] = (launch_job(gpu_id, BASELINE, *job), job)

    print(f"Launched {len(running)} jobs, {len(remaining)} remaining", flush=True)

    iteration = 0
    while remaining or running:
        iteration += 1

        # Check completed
        for gpu_id in list(running.keys()):
            (proc, start_time, log_fh), job = running[gpu_id]
            ret = proc.poll()
            if ret is not None:
                log_fh.close()
                elapsed = time.time() - start_time
                dataset, scene, method = job
                output_dir = os.path.join(RESULTS_BASE, BASELINE, dataset, scene, method)

                # Save training summary
                summary = {
                    "baseline": BASELINE, "scene": scene, "method": method,
                    "gpu": gpu_id, "exit_code": ret,
                    "wall_time_s": elapsed, "wall_time_min": elapsed / 60,
                }
                with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)

                if ret == 0:
                    print(f"  [GPU {gpu_id}] DONE: {scene}/{method} in {elapsed/60:.1f} min", flush=True)
                    completed.append((job, elapsed))
                else:
                    print(f"  [GPU {gpu_id}] FAIL: {scene}/{method} exit={ret}", flush=True)
                    failed.append((job, elapsed, ret))
                del running[gpu_id]

        # Launch new jobs
        busy = get_busy_gpus()
        for gpu_id in GPU_LIST:
            if gpu_id not in running and gpu_id not in busy and remaining:
                job = remaining.pop(0)
                running[gpu_id] = (launch_job(gpu_id, BASELINE, *job), job)

        if running:
            if iteration % 10 == 0:
                print(f"  [status] running={len(running)}, remaining={len(remaining)}, "
                      f"done={len(completed)}, failed={len(failed)}", flush=True)
            time.sleep(30)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  {BASELINE.upper()} TRAINING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Completed: {len(completed)}/{len(all_jobs)}", flush=True)
    print(f"  Failed: {len(failed)}", flush=True)

    summary = {
        "baseline": BASELINE,
        "total_jobs": len(all_jobs),
        "completed": len(completed),
        "failed": len(failed),
        "completed_jobs": [{"scene": j[1], "method": j[2], "elapsed_min": e/60} for j, e in completed],
        "failed_jobs": [{"scene": j[1], "method": j[2], "exit_code": r} for j, e, r in failed],
    }
    with open(os.path.join(RESULTS_BASE, BASELINE, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {RESULTS_BASE}/{BASELINE}/training_summary.json", flush=True)
