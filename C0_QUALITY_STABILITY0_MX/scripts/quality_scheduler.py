#!/usr/bin/env python3
"""C0-QUALITY-STABILITY0-MX scheduler.
Runs the 12 seed-expansion runs (seeds 43/44 x 3 scenes x {a0,c0}) on ONE
physical GPU, with the frozen preregistered alternating arm order, a pre-run
GPU idle guard, and a live foreign-pid contamination monitor.

Adapted verbatim in mechanism from the frozen causal_scheduler.py
(C0_SAME_PARENT_CAUSAL_LADDER_MX). Only differences: seeds 43/44, two arms
(a0/c0 only), frozen alternating order, /dev/shm runroot (pool 99% full).
"""
import argparse, json, os, subprocess, sys, time, csv

INCOMING = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived"
TRAINER = os.path.join(INCOMING, "mx_h8_trainer.py")
ENV_BIN = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin"
RUNROOT = "/dev/shm/c0_quality_stability/runs"

# FROZEN alternating order (goal section 15), same-GPU-per-pair
STREAMS = [
    {"stream": "s43_room",    "seed": 43, "scene": "room",    "gpu": 6, "arm_order": ["a0", "c0"]},
    {"stream": "s43_bicycle", "seed": 43, "scene": "bicycle", "gpu": 7, "arm_order": ["c0", "a0"]},
    {"stream": "s43_garden",  "seed": 43, "scene": "garden",  "gpu": 6, "arm_order": ["a0", "c0"]},
    {"stream": "s44_room",    "seed": 44, "scene": "room",    "gpu": 7, "arm_order": ["c0", "a0"]},
    {"stream": "s44_bicycle", "seed": 44, "scene": "bicycle", "gpu": 6, "arm_order": ["a0", "c0"]},
    {"stream": "s44_garden",  "seed": 44, "scene": "garden",  "gpu": 7, "arm_order": ["c0", "a0"]},
]
SEEDS = (43, 44)


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gpu_query(gpu):
    q = "index,uuid,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,memory.total,utilization.gpu"
    out = subprocess.run(["nvidia-smi", f"--id={gpu}", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    p = [x.strip() for x in out.split(",")]
    keys = ["index", "uuid", "name", "temp_c", "clocks_sm", "clocks_mem", "power_w",
            "mem_used_mib", "mem_total_mib", "util_pct"]
    return dict(zip(keys, p))


def compute_apps(gpu):
    uuid = gpu_query(gpu)["uuid"]
    out = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    apps = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4 and parts[0] == uuid:
            apps.append({"gpu_uuid": parts[0], "pid": int(parts[1]), "used_mib": parts[2], "name": parts[3]})
    return apps


def wait_gpu_idle(gpu, timeout_s=3600, poll_s=20):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        apps = compute_apps(gpu)
        if not apps:
            return None
        time.sleep(poll_s)
    return compute_apps(gpu)


def launch(gpu, arm, scene, seed, outdir):
    cmd = [f"{ENV_BIN}/python", TRAINER, "--arm", arm, "--scene", scene,
           "--iterations", "30000", "--seed", str(seed), "--gpu", str(gpu),
           "--lpips", "--checkpoints-lite", "--timing-grade", "PUBLICATION",
           "--final-eval", "all", "--outdir", outdir]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TMPDIR", "/dev/shm/c0_quality_stability/tmp")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/dev/shm/c0_quality_stability/torch_ext")
    env.setdefault("CUDA_CACHE_PATH", "/dev/shm/c0_quality_stability/cuda_cache")
    return subprocess.Popen(cmd, cwd=INCOMING, env=env), env


def monitor(proc, gpu, outdir):
    samples_path = os.path.join(outdir, "gpu_samples.csv")
    allowed = {proc.pid}
    rec = {"gpu_uuid": gpu_query(gpu)["uuid"], "n_samples": 0, "foreign_pids": [],
           "foreign_samples": [], "contaminated": False, "mem_used_mib": {"max": None},
           "util_pct": {"max": None}, "temp_c": {"max": None}}
    with open(samples_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ts", "util_pct", "mem_used_mib", "temp_c", "n_apps", "foreign_pids"])
        while proc.poll() is None:
            apps = compute_apps(gpu)
            foreign = [a for a in apps if a["pid"] not in allowed]
            g = gpu_query(gpu)
            w.writerow([utcnow(), g["util_pct"], g["mem_used_mib"], g["temp_c"], len(apps),
                        ",".join(str(a["pid"]) for a in foreign)])
            f.flush()
            rec["n_samples"] += 1
            for k, key in (("util_pct", "util_pct"), ("mem_used_mib", "mem_used_mib"), ("temp_c", "temp_c")):
                try:
                    v = float(g[key]); cur = rec[k]["max"]
                    rec[k]["max"] = v if cur is None else max(cur, v)
                except Exception:
                    pass
            if foreign:
                rec["foreign_pids"].extend([a["pid"] for a in foreign])
                rec["foreign_samples"].append({"ts": utcnow(), "apps": foreign})
                rec["contaminated"] = True
            time.sleep(15)
    rec["foreign_pids"] = sorted(set(rec["foreign_pids"]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="s43_room,s43_bicycle,s43_garden,s44_room,s44_bicycle,s44_garden")
    ap.add_argument("--out", default="/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX/RUN_INDEX_SEED4344.json")
    args = ap.parse_args()
    os.makedirs(RUNROOT, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("/dev/shm/c0_quality_stability", exist_ok=True)
    want = set(args.streams.split(","))
    index = {"campaign": "C0_QUALITY_STABILITY0_MX", "seeds": list(SEEDS), "runs": []}
    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    for st in STREAMS:
        if st["stream"] not in want:
            continue
        seed, scene, gpu = st["seed"], st["scene"], st["gpu"]
        for arm in st["arm_order"]:
            outdir = os.path.join(RUNROOT, f"{scene}_s{seed}_{arm}")
            os.makedirs(outdir, exist_ok=True)
            snap = gpu_query(gpu)
            print(f"[{utcnow()}] {scene}/s{seed}/{arm}: GPU{gpu} pre-snapshot util={snap['util_pct']}% mem={snap['mem_used_mib']}MiB; waiting idle", flush=True)
            foreign = wait_gpu_idle(gpu)
            if foreign is not None:
                print(f"[{utcnow()}] {scene}/s{seed}/{arm}: GPU{gpu} never cleared: {foreign}", flush=True)
                index["runs"].append({"scene": scene, "seed": seed, "arm": arm, "gpu": gpu,
                                      "status": "GPU_NEVER_IDLE", "foreign": foreign})
                with open(args.out, "w") as f: json.dump(index, f, indent=2)
                continue
            print(f"[{utcnow()}] {scene}/s{seed}/{arm}: launching 30K on GPU{gpu}", flush=True)
            proc, env = launch(gpu, arm, scene, seed, outdir)
            t0 = time.time()
            mon = monitor(proc, gpu, outdir)
            rc = proc.wait()
            wall = time.time() - t0
            rec = {"scene": scene, "seed": seed, "arm": arm, "gpu": gpu, "rc": rc,
                   "wall_s": wall, "outdir": outdir, "contaminated": mon["contaminated"],
                   "foreign_pids": mon["foreign_pids"], "gpu_snapshot": mon}
            rs = os.path.join(outdir, "results.json")
            if os.path.exists(rs):
                r = json.load(open(rs))
                rec["trainer_total_wall_s"] = r.get("timing", {}).get("total_wall_s")
                rec["final_N"] = r.get("final_N"); rec["initial_N"] = r.get("initial_N")
                rec["final_eval"] = r.get("final_eval")
                rec["phase_ms"] = r.get("timing", {}).get("phase_ms")
                rec["status"] = r.get("final_status", {}).get("status")
            else:
                rec["status"] = "NO_RESULTS"
            index["runs"].append(rec)
            print(f"[{utcnow()}] {scene}/s{seed}/{arm}: done rc={rc} wall={wall/60:.1f}min N={rec.get('final_N')} contam={mon['contaminated']}", flush=True)
            with open(args.out, "w") as f:
                json.dump(index, f, indent=2)
    print(f"[{utcnow()}] requested streams complete", flush=True)


if __name__ == "__main__":
    main()
