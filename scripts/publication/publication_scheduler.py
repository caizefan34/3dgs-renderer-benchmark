#!/usr/bin/env python3
"""PUBLICATION-PHASE scheduler: dynamic clean-GPU watching (same frozen
scheduling rule as FINAL-30K, Addendum B).

Waves (priority order, per publication plan §23):
  W1  P1 B1 gate:  b1 x {room, bicycle, garden}          (B1_REPRODUCTION_PASS gate)
  W2  P2 Stage A:  a0/a1/a2 x {room, bicycle, garden}    (cumulative ablation)
  W3  P4 corners:  c0e01 (c0, eps2d=0.1) x 3 + b1ae03 (b1a, eps2d=0.3) x 3

Scheduling rule (frozen, identical to FINAL-30K):
  - re-scan all 8 GPUs before every launch; CLEAN = zero visible compute
    processes AND memory.used < 500 MiB in two scans >= 120 s apart
  - never touch a GPU hosting any foreign process; one run per GPU
  - --timing-grade PUBLICATION only for verified-clean launches;
    mid-run foreign appearance => sticky contamination flag (downgrade, never drop)
  - idempotent resume (results.json skip), /proc adoption of live children
Run identity = <key>_<scene>; the trainer receives --arm + optional --eps2d.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
TRAINER = "/home/liaoyuanjun/3dgs-renderer-benchmark/publication_trainer.py"
OUT = "/mnt/storage_pool/liaoyuanjun/pub_runs"
ENV_BASE = {"PATH": "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/usr/bin:/bin",
            "HOME": os.path.expanduser("~")}

GATE_SCENES = ["room", "bicycle", "garden"]

# Optional file-driven queue: if present, it REPLACES the built-in queue and is
# re-read every poll cycle, so the queue can be extended (Stage B, P6, P5...)
# without restarting the scheduler or touching live runs. Entries:
#   {"key": "a0", "arm": "a0", "scene": "room", "extra": ["--eps2d", "0.3"]}
QUEUE_FILE = "/mnt/storage_pool/liaoyuanjun/pubphase/pub_queue.json"

BUILTIN_QUEUE = []
# W1: P1 B1 gate
for _s in GATE_SCENES:
    BUILTIN_QUEUE.append({"key": "b1", "arm": "b1", "scene": _s, "extra": []})
# W1b: P1 B0 gate (ORIGINAL_METHOD_PROTOCOL, wired + functional-gate-passed)
for _s in GATE_SCENES:
    BUILTIN_QUEUE.append({"key": "b0", "arm": "b0", "scene": _s, "extra": []})
# W2: P2 Stage A (a0/a1/a2)
for _s in GATE_SCENES:
    for _k in ("a0", "a1", "a2"):
        BUILTIN_QUEUE.append({"key": _k, "arm": _k, "scene": _s, "extra": []})
# W3: P4 eps2d corners
for _s in GATE_SCENES:
    BUILTIN_QUEUE.append({"key": "c0e01", "arm": "c0", "scene": _s, "extra": ["--eps2d", "0.1"]})
for _s in GATE_SCENES:
    BUILTIN_QUEUE.append({"key": "b1ae03", "arm": "b1a", "scene": _s, "extra": ["--eps2d", "0.3"]})


def load_queue():
    """File queue wins when present and well-formed; else the built-in queue."""
    try:
        with open(QUEUE_FILE) as f:
            q = json.load(f)
        out = []
        for e in q:
            out.append({"key": str(e["key"]), "arm": str(e["arm"]),
                        "scene": str(e["scene"]), "extra": [str(x) for x in e.get("extra", [])]})
        if out:
            return out
    except Exception:
        pass
    return BUILTIN_QUEUE


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=90).stdout
    except Exception:
        return ""


def _parse_mb(field):
    digits = "".join(ch for ch in field if ch.isdigit())
    return int(digits) if digits else 0


def scan_gpus():
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
    state_path = os.path.join(OUT, "pub_scheduler_state.json")
    st = load_state(state_path)
    st.setdefault("clean_seen_at", {})
    st.setdefault("contaminated", {})
    t0 = time.time()
    done_rids = {d["rid"] for d in st["done"]}
    failed_rids = {d["rid"] for d in st["failed"]}
    live_procs = {}

    log = open(os.path.join(OUT, "pub_scheduler.log"), "a", buffering=1)

    def L(msg):
        log.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg + "\n")

    for gpu, info in list(st["active"].items()):
        rid = info["rid"]
        if proc_alive(info["pid"]):
            L(f"adopted live run {rid} pid={info['pid']} gpu={gpu}")
        else:
            st["active"].pop(gpu)
            rd = os.path.join(OUT, rid)
            ok = os.path.exists(os.path.join(rd, "results.json"))
            rec = {"rid": rid, "arm": info["arm"], "scene": info["scene"],
                   "gpu": gpu, "rc": 0 if ok else 1, "note": "adopted-dead"}
            (st["done"] if ok else st["failed"]).append(rec)
            (done_rids if ok else failed_rids).add(rid)
            L(f"adopted-dead {rid} ok={ok}")

    while True:
        now = time.time()
        if now - t0 > args.max_total_hours * 3600:
            st["status"] = "INCOMPLETE_TIMEOUT"
            save_state(state_path, st)
            L("exit INCOMPLETE_TIMEOUT")
            return
        # ---- reap
        for gpu, info in list(st["active"].items()):
            rid = info["rid"]
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
            rd = os.path.join(OUT, rid)
            # own children carry a real rc; ADOPTED children (from a previous
            # scheduler incarnation) have rc=None -- judge them by results.json,
            # exactly like the adopted-dead startup path (else every adopted
            # run is falsely marked failed).
            ok = (rc == 0) if rc is not None else os.path.exists(
                os.path.join(rd, "results.json"))
            rec = {"rid": rid, "arm": info["arm"], "scene": info["scene"],
                   "gpu": gpu, "rc": rc if rc is not None else (0 if ok else 1),
                   "wall_s": round(now - info["started"], 1),
                   "timing_grade": info["timing_grade"],
                   "contaminated": bool(st["contaminated"].get(rid, False))}
            (st["done"] if ok else st["failed"]).append(rec)
            (done_rids if ok else failed_rids).add(rid)
            st["active"].pop(gpu, None)
            snapshot(f"{rid}_post")
            L(f"finished {rid} ok={ok} gpu={gpu} wall={rec['wall_s']}s "
              f"contaminated={rec['contaminated']}")
        # ---- queue (rid-keyed; active rids excluded)
        active_rids = {info["rid"] for info in st["active"].values()}
        remaining = [q for q in load_queue()
                     if f"{q['key']}_{q['scene']}" not in done_rids
                     and f"{q['key']}_{q['scene']}" not in failed_rids
                     and f"{q['key']}_{q['scene']}" not in active_rids]
        if not remaining and not st["active"]:
            st["status"] = "ALL_RUNS_COMPLETE" if not st["failed"] else "INCOMPLETE_WITH_FAILURES"
            save_state(state_path, st)
            L(f"exit {st['status']} done={len(st['done'])} failed={len(st['failed'])}")
            return
        # ---- scan + contamination watch
        procs = scan_gpus()
        for gpu, info in st["active"].items():
            rid = info["rid"]
            foreign = [p for p in procs.get(gpu, {}).get("pids", []) if p != info["pid"]]
            if foreign and rid not in st["contaminated"]:
                st["contaminated"][rid] = True
                L(f"CONTAMINATION {rid} gpu={gpu} foreign_pids={foreign}")
                snapshot(f"{rid}_contaminated")
        # ---- clean-stability (two scans >= 120 s apart)
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
        # ---- launch (longest-clean first)
        for gk in sorted(prev, key=lambda k: prev[k]):
            g = int(gk)
            if now - prev[gk] < 120:
                continue
            if not remaining:
                break
            entry = remaining.pop(0)
            rid = f"{entry['key']}_{entry['scene']}"
            rd = os.path.join(OUT, rid)
            if os.path.exists(os.path.join(rd, "results.json")):
                st["done"].append({"rid": rid, "arm": entry["arm"], "scene": entry["scene"],
                                   "gpu": g, "rc": 0, "note": "pre-existing results.json"})
                done_rids.add(rid)
                L(f"skip {rid} (results.json present)")
                prev.pop(gk, None)
                continue
            os.makedirs(rd, exist_ok=True)
            free_gb = shutil.disk_usage(OUT).free / (1024 ** 3)
            if free_gb < 10.0:
                L(f"HOLD {rid}: only {free_gb:.1f} GB free on {OUT} (need >= 10)")
                remaining.insert(0, entry)
                break
            snapshot(f"{rid}_pre")
            env = dict(ENV_BASE)
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            lfh = open(os.path.join(OUT, f"log_{rid}.txt"), "w")
            cmd = [PY, TRAINER, "--arm", entry["arm"], "--scene", entry["scene"],
                   "--iterations", "30000", "--outdir", rd, "--gpu", "0",
                   "--final-eval", "all", "--lpips",
                   "--timing-grade", "PUBLICATION"] + entry["extra"]
            proc = subprocess.Popen(cmd, env=env, stdout=lfh, stderr=subprocess.STDOUT,
                                    cwd="/home/liaoyuanjun/3dgs-renderer-benchmark")
            st["active"][str(g)] = {"rid": rid, "arm": entry["arm"], "scene": entry["scene"],
                                    "pid": proc.pid, "started": now,
                                    "timing_grade": "PUBLICATION"}
            live_procs[g] = (proc, lfh, rid)
            st["launched_any"] = True
            prev.pop(gk, None)
            L(f"LAUNCH {rid} gpu={g} pid={proc.pid} (verified clean at launch)"
              f"{' extra=' + ' '.join(entry['extra']) if entry['extra'] else ''}")
        save_state(state_path, st)
        L(f"poll: clean={sorted(now_clean_keys)} active={sorted(st['active'])} "
          f"queue={len(remaining)} done={len(st['done'])} failed={len(st['failed'])}")
        if not st["launched_any"] and not st["active"] and now - t0 > args.max_wait_hours * 3600:
            st["status"] = "PUB_READY_WAITING_FOR_CLEAN_GPU"
            save_state(state_path, st)
            L("exit PUB_READY_WAITING_FOR_CLEAN_GPU")
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
