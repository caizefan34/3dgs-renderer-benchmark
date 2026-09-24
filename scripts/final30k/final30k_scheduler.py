#!/usr/bin/env python3
"""FINAL-30K 26-run scheduler with dynamic clean-GPU watching (Addendum B).

Queue: 13 scenes x {b1a, c0_v3_final30k} = 26 runs of 30K iterations.
Scheduling rule (frozen):
  - Before EVERY launch, re-scan all 8 GPUs (never assume prior state).
  - CLEAN_PUBLICATION_GPU := GPU index with ZERO compute processes in TWO
    consecutive scans >= 120 s apart (stability guard against restart races).
  - Never touch a GPU that hosts any foreign process.
  - Launch the next pending run on a clean GPU; at most one run per GPU.
  - A run launched on a verified-clean GPU gets --timing-grade PUBLICATION;
    if a foreign process appears on its GPU mid-run, the run completes but is
    marked contaminated in the scheduler state (timing downgraded downstream).
  - Contamination snapshot before every launch and at completion.
  - Idempotent resume: runs with an existing results.json are skipped;
    live runs from a previous scheduler incarnation are adopted via /proc.
Exit states (scheduler_state.json.status):
  - ALL_RUNS_COMPLETE: queue drained, all runs produced results.json.
  - INCOMPLETE_WITH_FAILURES: some runs failed.
  - FINAL30K_READY_WAITING_FOR_CLEAN_GPU: max-wait-hours elapsed before any
    clean GPU ever appeared (no run launched).
  - INCOMPLETE_TIMEOUT: max-total-hours exceeded with work remaining.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
TRAINER = "/home/liaoyuanjun/3dgs-renderer-benchmark/final30k_trainer.py"
OUT = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
SCENES = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
          "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]
ARMS = ["b1a", "c0"]
ENV_BASE = {"PATH": "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/usr/bin:/bin",
            "HOME": os.path.expanduser("~")}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=90).stdout
    except Exception:
        # a failed/timed-out scan must behave as "no information": no GPU can
        # be declared clean from a missing scan (conservative failure mode)
        return ""


def _parse_mb(field):
    """'37,871 MiB' -> 37871 (nvidia-smi CSV uses thousands separators >= 10 GB)."""
    digits = "".join(ch for ch in field if ch.isdigit())
    return int(digits) if digits else 0


def scan_gpus():
    """Return {gpu_index: {"pids": [...], "used_mb": int}}.

    NOTE: compute-apps PIDs of OTHER users (e.g. root) are NOT visible under
    this UID, so a zero-PID scan is NOT sufficient evidence of a clean GPU.
    memory.used is visible for all GPUs and is the tamper-proof signal: a
    CLEAN GPU must show < 500 MiB used AND zero visible compute processes.
    """
    idx_uuid = {}
    for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx_uuid[parts[1]] = parts[0]
    procs = {i: {"pids": [], "used_mb": 0} for i in idx_uuid.values()}
    for line in sh("nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] in idx_uuid:
            procs[idx_uuid[parts[0]]]["used_mb"] = _parse_mb(",".join(parts[1:]))
    out = sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader")
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] in idx_uuid:
            try:
                procs[idx_uuid[parts[0]]]["pids"].append(int(parts[1]))
            except ValueError:
                pass
    return procs


def snapshot(name):
    d = os.path.join(OUT, ".snapshots")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.snap.txt"), "w") as f:
        f.write(time.strftime("== %Y-%m-%dT%H:%M:%S ==\n"))
        f.write(sh("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"))
        f.write(sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader"))


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"done": [], "failed": [], "active": {}, "launched_any": False,
            "clean_seen_at": {}, "contaminated": {}}


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, path)


def proc_alive(pid):
    """True only for a LIVE process. Zombies (dead, unreaped) must count as
    DEAD: os.kill(pid, 0) succeeds on a zombie, which would keep a finished
    run 'active' forever and stall the whole queue."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().split(") ")[-1].split()[0]
        return state != "Z"
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wait-hours", type=float, default=48.0)
    ap.add_argument("--max-total-hours", type=float, default=96.0)
    ap.add_argument("--poll-s", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    state_path = os.path.join(OUT, "scheduler_state.json")
    st = load_state(state_path)
    st.setdefault("clean_seen_at", {})
    st.setdefault("contaminated", {})
    t0 = time.time()
    finished = {(d["arm"], d["scene"]) for d in st["done"]}
    failed = {(d["arm"], d["scene"]) for d in st["failed"]}
    # in-memory Popen handles for children THIS process launched
    live_procs = {}  # gpu_index -> (Popen, logfile_handle, run_key)

    log = open(os.path.join(OUT, "scheduler.log"), "a", buffering=1)

    def L(msg):
        log.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg + "\n")

    # adopt live children from a previous scheduler incarnation (via /proc)
    for gpu, info in list(st["active"].items()):
        if proc_alive(info["pid"]):
            L(f"adopted live run {info['arm']}/{info['scene']} pid={info['pid']} gpu={gpu}")
        else:
            st["active"].pop(gpu)
            rd = os.path.join(OUT, f"{info['arm']}_{info['scene']}")
            ok = os.path.exists(os.path.join(rd, "results.json"))
            rec = {"arm": info["arm"], "scene": info["scene"], "gpu": gpu,
                   "rc": 0 if ok else 1, "note": "adopted-dead"}
            (st["done"] if ok else st["failed"]).append(rec)
            (finished if ok else failed).add((info["arm"], info["scene"]))
            L(f"adopted-dead {info['arm']}/{info['scene']} ok={ok}")

    while True:
        now = time.time()
        if now - t0 > args.max_total_hours * 3600:
            st["status"] = "INCOMPLETE_TIMEOUT"
            save_state(state_path, st)
            L("exit INCOMPLETE_TIMEOUT")
            return
        # ---- reap finished children (own children via poll, adopted via /proc)
        # NOTE: st["active"] keys are STRINGS ("0") while live_procs keys are
        # INTS — the lookup MUST convert, or own children are never reaped via
        # poll() and dead runs stall as zombies (bug fixed 2026-09-23).
        for gpu, info in list(st["active"].items()):
            key = f"{info['arm']}_{info['scene']}"
            if int(gpu) in live_procs:
                proc, lfh, _ = live_procs[int(gpu)]
                rc = proc.poll()
                alive = rc is None
            else:
                alive = proc_alive(info["pid"])
                rc = None
            if alive:
                continue
            if int(gpu) in live_procs:
                lfh.close()
                del live_procs[int(gpu)]
            rd = os.path.join(OUT, info["arm"] + "_" + info["scene"])
            ok = (rc == 0) and os.path.exists(os.path.join(rd, "results.json"))
            rec = {"arm": info["arm"], "scene": info["scene"], "gpu": gpu,
                   "rc": rc if rc is not None else (0 if ok else 1),
                   "wall_s": round(now - info["started"], 1),
                   "timing_grade": info["timing_grade"],
                   "contaminated": bool(st["contaminated"].get(key, False))}
            (st["done"] if ok else st["failed"]).append(rec)
            (finished if ok else failed).add((info["arm"], info["scene"]))
            st["active"].pop(gpu, None)
            snapshot(f"{key}_post")
            L(f"finished {key} ok={ok} gpu={gpu} wall={rec['wall_s']}s "
              f"contaminated={rec['contaminated']}")
        # ---- queue state (active pairs MUST be excluded: an in-flight run is
        # still absent from finished/failed and would otherwise be re-launched)
        active_pairs = {(info["arm"], info["scene"])
                        for info in st["active"].values()}
        remaining = [(a, s) for s in SCENES for a in ARMS
                     if (a, s) not in finished and (a, s) not in failed
                     and (a, s) not in active_pairs]
        if not remaining and not st["active"]:
            st["status"] = "ALL_RUNS_COMPLETE" if not st["failed"] else "INCOMPLETE_WITH_FAILURES"
            save_state(state_path, st)
            L(f"exit {st['status']} done={len(st['done'])} failed={len(st['failed'])}")
            return
        # ---- scan + contamination watch
        procs = scan_gpus()
        for gpu, info in st["active"].items():
            key = f"{info['arm']}_{info['scene']}"
            # Contamination := a foreign process visible to us on our GPU.
            # (Root-owned foreign processes are invisible to compute-apps under
            # this UID and cannot be separated from our own run's memory use;
            # they are excluded by the <500 MiB launch-time cleanliness check.)
            foreign = [p for p in procs.get(gpu, {}).get("pids", []) if p != info["pid"]]
            if foreign and key not in st["contaminated"]:
                st["contaminated"][key] = True
                L(f"CONTAMINATION {key} gpu={gpu} foreign_pids={foreign}")
                snapshot(f"{key}_contaminated")
        # ---- clean-stability tracking (two scans >= 120 s apart)
        prev = st["clean_seen_at"]
        now_clean = {g for g, v in procs.items()
                     if not v["pids"] and v["used_mb"] < 500
                     and str(g) not in st["active"]}
        now_clean_keys = {str(g) for g in now_clean}
        for g in list(prev):
            if g not in now_clean_keys:
                prev.pop(g)
        for g in now_clean:
            gk = str(g)
            if gk not in prev:
                prev[gk] = now
        # ---- launch on stable-clean GPUs (longest-clean first)
        for gk in sorted(prev, key=lambda k: prev[k]):
            g = int(gk)
            if now - prev[gk] < 120:
                continue
            if not remaining:
                break
            arm, scene = remaining.pop(0)
            rd = os.path.join(OUT, f"{arm}_{scene}")
            if os.path.exists(os.path.join(rd, "results.json")):
                st["done"].append({"arm": arm, "scene": scene, "gpu": g, "rc": 0,
                                   "note": "pre-existing results.json"})
                finished.add((arm, scene))
                L(f"skip {arm}/{scene} (results.json present)")
                prev.pop(gk, None)
                continue
            os.makedirs(rd, exist_ok=True)
            # Disk guard: the storage pool is near capacity (no checkpoints are
            # saved in this phase by design). Require >= 10 GB free before launch.
            free_gb = shutil.disk_usage(OUT).free / (1024 ** 3)
            if free_gb < 10.0:
                L(f"HOLD {arm}/{scene}: only {free_gb:.1f} GB free on {OUT} (need >= 10)")
                remaining.insert(0, (arm, scene))
                break
            snapshot(f"{arm}_{scene}_pre")
            env = dict(ENV_BASE)
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            lfh = open(os.path.join(OUT, f"log_{arm}_{scene}.txt"), "w")
            proc = subprocess.Popen(
                [PY, TRAINER, "--arm", arm, "--scene", scene,
                 "--iterations", "30000", "--outdir", rd, "--gpu", "0",
                 "--final-eval", "all", "--lpips",
                 "--timing-grade", "PUBLICATION"],
                env=env, stdout=lfh, stderr=subprocess.STDOUT,
                cwd="/home/liaoyuanjun/3dgs-renderer-benchmark")
            st["active"][str(g)] = {"arm": arm, "scene": scene, "pid": proc.pid,
                                    "started": now, "timing_grade": "PUBLICATION"}
            live_procs[g] = (proc, lfh, f"{arm}_{scene}")
            st["launched_any"] = True
            prev.pop(gk, None)
            L(f"LAUNCH {arm}/{scene} gpu={g} pid={proc.pid} (verified clean at launch)")
        save_state(state_path, st)
        L(f"poll: clean={sorted(now_clean_keys)} active={sorted(st['active'])} "
          f"queue={len(remaining)} done={len(st['done'])} failed={len(st['failed'])}")
        # ---- give-up: nothing ever launched and wait window expired
        if not st["launched_any"] and not st["active"] and now - t0 > args.max_wait_hours * 3600:
            st["status"] = "FINAL30K_READY_WAITING_FOR_CLEAN_GPU"
            save_state(state_path, st)
            L("exit FINAL30K_READY_WAITING_FOR_CLEAN_GPU")
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
