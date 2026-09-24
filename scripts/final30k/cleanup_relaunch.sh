#!/bin/bash
# Cleanup: single-scheduler truth injection + setsid relaunch (scan-v3 code).
# - kills EVERY scheduler instance (sweeps until zero)
# - injects the verified live orphan c0/bicycle (pid 2218700) into active
# - relaunches ONE scheduler via setsid (new session, immune to ssh teardown)
set -e
echo "== sweep schedulers =="
for i in 1 2 3; do
  pkill -f 'final30k_[s]cheduler.py' 2>/dev/null || true
  sleep 2
done
N=$(ps aux | grep 'final30k_[s]cheduler.py' | grep -v grep | wc -l)
echo "schedulers remaining: $N"
[ "$N" = "0" ] || { echo "FATAL: scheduler survived 3 pkill sweeps"; exit 1; }

echo "== inject truth =="
python3 - <<'EOF'
import json, os
p = "/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"
s = {"done": [], "failed": [], "active": {}, "launched_any": True,
     "clean_seen_at": {}, "contaminated": {}}
# c0/bonsai: completed 19:17:33->~19:35 on GPU 0 (results.json verified by
# the 19:44 adoption); recorded as done.
rd_bonsai = "/mnt/storage_pool/liaoyuanjun/final30k_runs/c0_bonsai"
assert os.path.exists(os.path.join(rd_bonsai, "results.json")), "c0_bonsai results.json missing!"
s["done"].append({"arm": "c0", "scene": "bonsai", "gpu": "0", "rc": 0,
                  "note": "completed 19:17:33-~19:35, adopted 19:44"})
# c0/bicycle: orphan launched 19:37:34 on GPU 0 by the previous incarnation.
pid = 2218700
rd = "/mnt/storage_pool/liaoyuanjun/final30k_runs/c0_bicycle"
alive = os.path.exists(f"/proc/{pid}")
has_results = os.path.exists(os.path.join(rd, "results.json"))
print(f"c0/bicycle pid={pid} alive={alive} results={has_results}")
if alive:
    s["active"]["0"] = {"arm": "c0", "scene": "bicycle", "pid": pid,
                        "started": 1790162254, "timing_grade": "PUBLICATION"}
elif has_results:
    s["done"].append({"arm": "c0", "scene": "bicycle", "gpu": "0", "rc": 0,
                      "note": "orphan finished before injection"})
else:
    s["failed"].append({"arm": "c0", "scene": "bicycle", "gpu": "0", "rc": 1,
                        "note": "orphan died without results; re-queued"})
json.dump(s, open(p, "w"), indent=2)
print("state:", json.dumps(s))
EOF

echo "== relaunch (setsid) =="
setsid nohup bash /mnt/storage_pool/liaoyuanjun/launch_scheduler.sh < /dev/null > /dev/null 2>&1 &
sleep 5
N=$(ps aux | grep 'final30k_[s]cheduler.py' | grep -v grep | wc -l)
echo "schedulers after relaunch: $N"
tail -2 /mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler.log
