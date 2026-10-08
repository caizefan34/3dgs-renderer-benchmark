#!/usr/bin/env python3
"""P12 paired-campaign scheduler (clean-A100 policy, back-to-back pairs).

Differences from the P8 run_group_scheduler.py:
- each group = ONE scene PAIR (2 jobs, preregistered alternating arm order);
  the pair runs back-to-back on ONE physical GPU with no other publication
  pair interleaved on that GPU;
- strict clean-GPU gate BEFORE each pair: N_CONSEC consecutive 30s checks
  with ZERO compute processes on the GPU (refuses co-tenant GPUs; waits);
- launcher meta records explicit ts_start/ts_end (epoch s) per job for the
  contamination join (no mtime inference);
- passes --loss fused to publication_trainer_p12.py (P12 strong baseline).

Contamination response is post-hoc (watchdog join): contaminated attempts
are preserved and the PAIR is rerun unchanged on a clean GPU.
"""
import argparse
import json
import os
import queue
import shlex
import subprocess
import threading
import time
import pathlib

POLL_S = 30
N_CONSEC = 10  # 10 consecutive clean checks x 30s = 5 min gate


def gpu_uuid_of(gpu):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return out


def compute_apps_on(uuid):
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip() and uuid in l]


def gpu_snapshot(gpu, path):
    cmd = ["nvidia-smi", "-i", str(gpu),
           "--query-gpu=timestamp,index,uuid,name,pstate,memory.used,utilization.gpu,"
           "power.draw,clocks.sm,clocks.mem,temperature.gpu",
           "--format=csv,noheader"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    pathlib.Path(path).write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")


def wait_clean(gpu, uuid, log):
    """Block until N_CONSEC consecutive clean checks. Returns wait seconds."""
    t0 = time.time()
    consec = 0
    while consec < N_CONSEC:
        procs = compute_apps_on(uuid)
        if procs:
            consec = 0
            log(f"[GPU{gpu}] clean-gate: foreign/other compute present "
                f"({procs[:2]}); waiting")
        else:
            consec += 1
        if consec < N_CONSEC:
            time.sleep(POLL_S)
    return time.time() - t0


def run_job(job, gpu, args, log):
    outdir = pathlib.Path(args.out) / job["run_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / "EPIC05_SUCCESS"
    if done.exists() and not args.rerun:
        log(f"SKIP {job['run_id']} (already done)")
        return True

    gpu_snapshot(gpu, outdir / "gpu_before.csv")
    cmd = [args.python, args.trainer,
           "--arm", job["arm"], "--scene", job["scene"],
           "--iterations", str(job.get("iterations", 30000)),
           "--outdir", str(outdir), "--gpu", "0",
           "--lpips", "--final-eval", "all", "--checkpoints",
           "--timing-grade", "PUBLICATION",
           "--eps2d", str(job.get("eps2d", 0.3)),
           "--loss", job.get("loss", "fused")]
    if job.get("seed", 42) != 42:
        cmd += ["--seed", str(job["seed"])]
    (outdir / "command.txt").write_text(
        " ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ts_start = time.time()
    t0 = time.time()
    with open(outdir / "stdout.log", "w", buffering=1) as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    ts_end = time.time()
    gpu_snapshot(gpu, outdir / "gpu_after.csv")
    meta = {"run_id": job["run_id"], "group_gpu_physical": gpu,
            "returncode": p.returncode,
            "ts_start": ts_start, "ts_end": ts_end,
            "wall_launcher_s": time.time() - t0, "job": job}
    (outdir / "epic05_launcher_meta.json").write_text(json.dumps(meta, indent=2),
                                                      encoding="utf-8")
    if p.returncode == 0 and (outdir / "results.json").exists():
        done.write_text("PASS\n")
        log(f"PASS {job['run_id']} rc=0 wall={meta['wall_launcher_s']/60:.1f}min")
        return True
    (outdir / "EPIC05_FAILED").write_text(f"returncode={p.returncode}\n")
    log(f"FAIL {job['run_id']} rc={p.returncode}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--trainer", default="/root/epic05_p12/publication_trainer_p12.py")
    ap.add_argument("--python", default="/root/epic05_py_wrapper.sh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    logf = open(os.path.join(args.out, "scheduler.log"), "a", buffering=1)

    def base_log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    groups_by_gpu = {}
    for g in m["groups"]:
        groups_by_gpu.setdefault(g["gpu"], []).append(g)

    failures = []
    lock = threading.Lock()

    def worker(gpu):
        uuid = gpu_uuid_of(gpu)
        base_log(f"[GPU{gpu}] worker start uuid={uuid} groups="
                 f"{[g['group_id'] for g in groups_by_gpu[gpu]]}")
        for g in groups_by_gpu[gpu]:
            wait_s = wait_clean(gpu, uuid, base_log)
            base_log(f"[GPU{gpu}] clean-gate passed ({wait_s:.0f}s) -> "
                     f"PAIR {g['group_id']} jobs={[j['run_id'] for j in g['jobs']]}")
            pair_start = time.time()
            ok = True
            for job in g["jobs"]:
                if not run_job(job, gpu, args, base_log):
                    ok = False
                    with lock:
                        failures.append(job["run_id"])
                    # do not run the pair partner after a failure
                    break
            pair_end = time.time()
            with open(os.path.join(args.out, f"{g['group_id']}.pairmeta.json"), "w") as f:
                json.dump({"group_id": g["group_id"], "gpu": gpu,
                           "ts_pair_start": pair_start, "ts_pair_end": pair_end,
                           "ok": ok}, f, indent=2)
            base_log(f"[GPU{gpu}] PAIR {g['group_id']} "
                     f"{'PASS' if ok else 'FAIL'} ({(pair_end-pair_start)/60:.1f}min)")
        base_log(f"[GPU{gpu}] worker done")

    threads = [threading.Thread(target=worker, args=(gpu,))
               for gpu in sorted(groups_by_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = {"failures": failures, "n_failures": len(failures),
               "campaign_id": m.get("campaign_id"),
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out, "scheduler_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    base_log(f"CAMPAIGN DONE failures={failures or 'none'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
