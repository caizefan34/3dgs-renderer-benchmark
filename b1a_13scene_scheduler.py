#!/usr/bin/env python3
"""Launch all 26 B1/B1A 13-scene training runs across the 7 free A100 GPUs.

Strategy: assign each (scene, method) job to a GPU queue. 7 free GPUs
(0,1,2,3,5,6,7). Each GPU processes its queue sequentially.
Output to /dev/shm/accutile30k/<scene>/<method>.
"""
import os, sys, time, json, subprocess

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
OUT_BASE = "/dev/shm/accutile30k"
TRAIN_SCRIPT = os.path.join(REPO, "b1a_13scene_train.py")
LOG_BASE = "/dev/shm/accutile30k_logs"

MIPNERF360 = ["bicycle", "bonsai", "counter", "flowers", "garden",
              "kitchen", "room", "stump", "treehill"]
TANKSTEMPLES = ["train", "truck"]
DEEPBLENDING = ["drjohnson", "playroom"]

SCENES = [(s, "mipnerf360") for s in MIPNERF360] + \
         [(s, "tanksandtemples") for s in TANKSTEMPLES] + \
         [(s, "deepblending") for s in DEEPBLENDING]

METHODS = ["b1", "b1a"]

# Free GPUs (GPU 4 is in use by another user)
GPUS = [0, 1, 2, 3, 5, 6, 7]


def is_done(scene, method):
    return os.path.exists(os.path.join(OUT_BASE, scene, method, "training_results.json"))


def get_busy_gpus():
    try:
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
            if uuid in uuid_to_idx:
                busy.add(uuid_to_idx[uuid])
        return busy
    except Exception:
        return set()


def launch(gpu_id, scene, dataset, method):
    output_dir = os.path.join(OUT_BASE, scene, method)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(LOG_BASE, scene)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{method}.log")
    cmd = ["python3", "-u", TRAIN_SCRIPT,
           "--scene", scene, "--method", method,
           "--gpu", str(gpu_id), "--output_dir", output_dir]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"  [GPU {gpu_id}] Launching: {scene}/{method}", flush=True)
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            cwd=REPO, env=env)
    return (scene, method, proc, log_fh)


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(LOG_BASE, exist_ok=True)

    # Build job list: all (scene, method) pairs not yet done.
    all_jobs = []
    for scene, dataset in SCENES:
        for method in METHODS:
            if is_done(scene, method):
                print(f"  SKIP (done): {scene}/{method}", flush=True)
            else:
                all_jobs.append((scene, dataset, method))

    print(f"\nJobs to launch: {len(all_jobs)}", flush=True)
    if not all_jobs:
        print("All jobs already complete.", flush=True)
        return

    running = {}   # gpu_id -> (scene, method, proc, log_fh, start_time)
    completed = []
    failed = []

    while all_jobs or running:
        # Reap finished
        for gpu_id in list(running.keys()):
            scene, method, proc, log_fh, start_time = running[gpu_id]
            ret = proc.poll()
            if ret is not None:
                log_fh.close()
                elapsed = time.time() - start_time
                if ret == 0:
                    print(f"  [GPU {gpu_id}] DONE: {scene}/{method} in {elapsed/60:.1f} min", flush=True)
                    completed.append((scene, method, elapsed))
                else:
                    print(f"  [GPU {gpu_id}] FAIL: {scene}/{method} exit={ret}", flush=True)
                    failed.append((scene, method, elapsed, ret))
                del running[gpu_id]

        # Launch on free GPUs
        for gpu_id in GPUS:
            if gpu_id not in running and all_jobs:
                scene, dataset, method = all_jobs.pop(0)
                running[gpu_id] = launch(gpu_id, scene, dataset, method) + (time.time(),)

        if running:
            time.sleep(30)

    print(f"\n=== ALL DONE ===", flush=True)
    print(f"Completed: {len(completed)}, Failed: {len(failed)}", flush=True)
    for scene, method, elapsed in completed:
        print(f"  OK: {scene}/{method}: {elapsed/60:.1f} min", flush=True)
    for scene, method, elapsed, ret in failed:
        print(f"  FAIL: {scene}/{method}: exit={ret}", flush=True)


if __name__ == "__main__":
    main()
