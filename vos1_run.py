import os, sys, json, subprocess, threading, queue, time

PY = "/root/miniforge3/envs/p16_fastergs/bin/python"
TRAIN = "/root/3dgs-renderer-benchmark/vos1_train.py"
SRC = "/root/epic05_p16rb/common_data/mipnerf360"
OUT = "/root/vos1_out/results"
os.makedirs(OUT, exist_ok=True)

SCENES = ["room", "garden", "bicycle"]
KS = [1, 2, 4, 8]
NGPU = 8

jobs = []
for scene in SCENES:
    for k in KS:
        jobs.append((scene, k))

q = queue.Queue()
for j in jobs:
    q.put(j)

results = {}
lock = threading.Lock()
done = [0]

def worker(gpu):
    while True:
        try:
            scene, k = q.get_nowait()
        except queue.Empty:
            return
        tag = f"{scene}_k{k}"
        model_path = os.path.join(OUT, "models", tag)
        result_path = os.path.join(OUT, f"result_{tag}.json")
        trace_path = os.path.join(OUT, f"trace_{tag}.json")
        os.makedirs(model_path, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [PY, TRAIN,
               "--scene_name", scene,
               "--source_path", os.path.join(SRC, scene),
               "--model_path", model_path,
               "--k", str(k),
               "--iterations", "30000",
               "--result_path", result_path,
               "--trace_path", trace_path]
        t0 = time.time()
        print(f"[GPU{gpu}] START {tag} (t={time.strftime('%H:%M:%S')})", flush=True)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"[GPU{gpu}] FAIL {tag} rc={proc.returncode} ({dt:.0f}s)", flush=True)
            print("STDOUT:", proc.stdout[-2000:], flush=True)
            print("STDERR:", proc.stderr[-3000:], flush=True)
        else:
            print(f"[GPU{gpu}] DONE  {tag} ({dt:.0f}s)", flush=True)
        with lock:
            done[0] += 1
        q.task_done()

threads = []
for gpu in range(NGPU):
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()

print("ALL_JOBS_DONE", flush=True)
