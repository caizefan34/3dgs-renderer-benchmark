#!/usr/bin/env python3
"""CAUSAL ladder scheduler: runs the 4-arm same-parent ladder for ONE scene
on ONE physical GPU, with a pre-run GPU guard and live contamination monitor.

Preregistered arm order per scene (section 7):
  room    : a0 -> a1 -> a2 -> c0
  bicycle : a2 -> c0 -> a0 -> a1
  garden  : a1 -> a0 -> c0 -> a2
(seed 42 for all Phase-1 runs)
"""
import argparse, json, os, subprocess, sys, time, csv

INCOMING = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived"
TRAINER = os.path.join(INCOMING, "mx_h8_trainer.py")
ENV_BIN = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin"
PREREG_ORDER = {"room": ["a0", "a1", "a2", "c0"],
                "bicycle": ["a2", "c0", "a0", "a1"],
                "garden": ["a1", "a0", "c0", "a2"]}
SEED = 42


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


def wait_gpu_idle(gpu, timeout_s=1800, poll_s=20):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        apps = compute_apps(gpu)
        if not apps:
            return None
        time.sleep(poll_s)
    return compute_apps(gpu)


def launch(gpu, arm, scene, outdir):
    cmd = [f"{ENV_BIN}/python", TRAINER, "--arm", arm, "--scene", scene,
           "--iterations", "30000", "--seed", str(SEED), "--gpu", str(gpu),
           "--lpips", "--checkpoints-lite", "--timing-grade", "PUBLICATION",
           "--final-eval", "all", "--outdir", outdir]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TMPDIR", "/dev/shm/causal_ladder/tmp")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/dev/shm/causal_ladder/torch_ext")
    env.setdefault("CUDA_CACHE_PATH", "/dev/shm/causal_ladder/cuda_cache")
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
            w.writerow([utcnow(), g["util_pct"], g["mem_used_mib"], g["temp_c"], len(apps), ",".join(str(a["pid"]) for a in foreign)])
            f.flush()
            rec["n_samples"] += 1
            for k, key in (("util_pct", "util_pct"), ("mem_used_mib", "mem_used_mib"), ("temp_c", "temp_c")):
                try:
                    v = float(g[key])
                    cur = rec[k]["max"]
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
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--runroot", default="/dev/shm/causal_ladder/runs")
    ap.add_argument("--out", default=os.path.join(INCOMING, "causal_out", "CAUSAL_RUN_INDEX.json"))
    args = ap.parse_args()

    os.makedirs(args.runroot, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    arms = PREREG_ORDER[args.scene]
    index = {"scene": args.scene, "seed": SEED, "gpu": args.gpu, "prereg_order": arms, "runs": []}

    for arm in arms:
        outdir = os.path.join(args.runroot, f"{args.scene}_s{SEED}_{arm}")
        os.makedirs(outdir, exist_ok=True)
        print(f"[{utcnow()}] {args.scene}/{arm}: waiting for GPU {args.gpu} idle", flush=True)
        foreign = wait_gpu_idle(args.gpu)
        if foreign is not None:
            print(f"[{utcnow()}] {args.scene}/{arm}: GPU {args.gpu} never cleared: {foreign}", flush=True)
            index["runs"].append({"arm": arm, "scene": args.scene, "status": "GPU_NEVER_IDLE", "foreign": foreign})
            continue
        print(f"[{utcnow()}] {args.scene}/{arm}: launching 30K", flush=True)
        proc, env = launch(args.gpu, arm, args.scene, outdir)
        t0 = time.time()
        mon = monitor(proc, args.gpu, outdir)
        rc = proc.wait()
        wall = time.time() - t0
        rec = {"arm": arm, "scene": args.scene, "seed": SEED, "gpu": args.gpu,
               "rc": rc, "wall_s": wall, "outdir": outdir,
               "contaminated": mon["contaminated"], "foreign_pids": mon["foreign_pids"],
               "gpu_snapshot": mon}
        rs_path = os.path.join(outdir, "results.json")
        if os.path.exists(rs_path):
            r = json.load(open(rs_path))
            rec["trainer_total_wall_s"] = r.get("timing", {}).get("total_wall_s")
            rec["final_N"] = r.get("final_N")
            rec["initial_N"] = r.get("initial_N")
            rec["final_eval"] = r.get("final_eval")
            rec["phase_ms"] = r.get("timing", {}).get("phase_ms")
            rec["status"] = r.get("final_status", {}).get("status")
        else:
            rec["status"] = "NO_RESULTS"
        index["runs"].append(rec)
        print(f"[{utcnow()}] {args.scene}/{arm}: done rc={rc} wall={wall/60:.1f}min N={rec.get('final_N')}", flush=True)
        with open(args.out, "w") as f:
            json.dump(index, f, indent=2)
    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[{utcnow()}] {args.scene} ladder complete")


if __name__ == "__main__":
    main()
