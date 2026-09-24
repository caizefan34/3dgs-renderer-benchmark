#!/bin/bash
# fix3_rollout.sh — zombie-reap bug fix + authoritative state surgery.
#
# Forensic basis (mx scheduler.log + file mtimes, 2026-09-23):
#   c0/bonsai    done CLEAN    (19:17-19:35, gpu0)
#   c0/bicycle   done CLEAN    (19:37-20:05, gpu0)
#   b1a/bicycle  done CLEAN    (orphan attempt 20:05:49-20:33:22, gpu0;
#                               first attempt 19:37-20:03 was contaminated,
#                               its files were overwritten by the clean attempt)
#   c0/counter   done CLEAN    (attempt2 20:39:16-20:57:40, gpu0; attempt1
#                               19:17-19:44 was contaminated, overwritten)
#   b1a/drjohnson done CLEAN   (20:45:19-21:04:30, gpu1)
#   b1a/counter  done CONTAMINATED (20:36:11-21:11:03, gpu7; root pid 2622247
#                               from 20:40:17; wall inflated 33.9 min)
#   b1a/bonsai   CRASHED (OOM at 20:40 root invasion) -> re-queue, no record
#   c0/drjohnson done CLEAN (21:09:33-21:27, gpu2, PSNR 30.19)
set -e
echo "== stop scheduler =="
for i in 1 2 3; do pkill -f 'final30k_[s]cheduler.py' 2>/dev/null || true; sleep 2; done
N=$(ps aux | grep 'final30k_[s]cheduler.py' | grep -v grep | wc -l)
echo "schedulers remaining: $N"
[ "$N" = "0" ] || { echo "FATAL: scheduler survived sweeps"; exit 1; }

echo "== verify completions on disk =="
python3 - <<'EOF'
import os, sys
for run in ("c0_bonsai", "c0_bicycle", "b1a_bicycle", "c0_counter",
            "b1a_drjohnson", "b1a_counter", "c0_drjohnson"):
    if not os.path.exists(f"/mnt/storage_pool/liaoyuanjun/final30k_runs/{run}/results.json"):
        print(f"FATAL: {run} results.json missing"); sys.exit(1)
print("all 7 completed-run results.json present")
EOF

echo "== state surgery =="
python3 - <<'EOF'
import json
p = "/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"
s = json.load(open(p))
s["done"] = [
    {"arm": "c0", "scene": "bonsai", "gpu": "0", "rc": 0, "note": "clean 19:17-19:35"},
    {"arm": "c0", "scene": "bicycle", "gpu": "0", "rc": 0, "note": "clean 19:37-20:05"},
    {"arm": "b1a", "scene": "bicycle", "gpu": "0", "rc": 0,
     "note": "clean orphan attempt 20:05:49-20:33:22 (contaminated first attempt overwritten)"},
    {"arm": "c0", "scene": "counter", "gpu": "0", "rc": 0,
     "note": "clean attempt2 20:39:16-20:57:40 (contaminated attempt1 overwritten)"},
    {"arm": "b1a", "scene": "drjohnson", "gpu": "1", "rc": 0, "note": "clean 20:45:19-21:04:30"},
    {"arm": "c0", "scene": "drjohnson", "gpu": "2", "rc": 0, "note": "clean 21:09:33-21:27"},
    {"arm": "b1a", "scene": "counter", "gpu": "7", "rc": 0, "contaminated": True,
     "note": "completed CONTAMINATED 20:36:11-21:11:03 (root 2622247 from 20:40:17) - retry candidate"},
]
s["failed"] = []
# b1a/bonsai: OOM-crashed attempt discarded; pair re-enters the queue
s["active"] = {}
s["contaminated"] = {"b1a_counter": True}
s["contaminated_attempts"] = [
    {"arm": "b1a", "scene": "bicycle", "attempt": 1,
     "note": "19:37-20:03 gpu3 contaminated 19:41:35; overwritten by clean orphan attempt"},
    {"arm": "c0", "scene": "counter", "attempt": 1,
     "note": "19:17-19:44 gpu3 contaminated 19:22:35; overwritten by clean attempt2"},
    {"arm": "b1a", "scene": "bonsai", "attempt": 1,
     "note": "20:36-20:40 gpu3 contaminated 20:40:16 then OOM crash; discarded, re-queued"},
    {"arm": "b1a", "scene": "counter", "attempt": 2,
     "note": "20:36:11-21:11:03 gpu7 completed contaminated (root 2622247); preserved, retry pending"},
]
s["launched_any"] = True
s["clean_seen_at"] = {}
json.dump(s, open(p, "w"), indent=2)
n = len(s["done"])
print(f"state: done={n} active=0 queue={26 - n} contaminated={{'b1a_counter': True}}")
EOF

echo "== relaunch (fixed zombie-reap code) =="
setsid nohup bash /mnt/storage_pool/liaoyuanjun/launch_scheduler.sh < /dev/null > /dev/null 2>&1 &
sleep 6
N=$(ps aux | grep 'final30k_[s]cheduler.py' | grep -v grep | wc -l)
echo "schedulers after relaunch: $N"
tail -3 /mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler.log
