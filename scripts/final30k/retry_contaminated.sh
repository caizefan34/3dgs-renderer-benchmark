#!/bin/bash
# retry_contaminated.sh — clean-retry pass for contamination-flagged runs.
#
# PROTOCOL: a run whose GPU was invaded by a foreign process is an invalid
# timing measurement. The frozen artifacts demand a clean measurement; the
# contaminated attempt is PRESERVED (moved to *_contaminated_attempt<N>) and
# disclosed, never deleted. This script re-queues those pairs so the scheduler
# relaunches them when a verified-clean GPU appears.
#
# SAFETY: refuses to run unless the scheduler is STOPPED and no runs are
# active (state.active empty). Run this only between scheduler incarnations.
set -e

STATE=/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json
RUNS=/mnt/storage_pool/liaoyuanjun/final30k_runs

echo "== preconditions =="
if pgrep -f 'final30k_[s]cheduler.py' > /dev/null; then
  echo "FATAL: scheduler is running — stop it first (no active-run disturbance)"; exit 1
fi
python3 - <<'EOF'
import json, sys
s = json.load(open("/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"))
if s["active"]:
    print("FATAL: active runs present:", s["active"]); sys.exit(1)
EOF

echo "== retry surgery =="
python3 - <<'EOF'
import json, os, shutil, glob

STATE = "/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"
RUNS = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
s = json.load(open(STATE))
s.setdefault("contaminated_attempts", [])

retried = []
for key in sorted(s.get("contaminated", {})):
    arm, scene = key.split("_", 1)
    rd = os.path.join(RUNS, f"{arm}_{scene}")
    if not os.path.exists(os.path.join(rd, "results.json")):
        print(f"  skip {key}: no results.json (run still incomplete?)")
        continue
    # next attempt index
    n = 1
    while os.path.exists(f"{rd}_contaminated_attempt{n}"):
        n += 1
    dst = f"{rd}_contaminated_attempt{n}"
    shutil.move(rd, dst)
    print(f"  moved {rd} -> {dst}")
    s["contaminated_attempts"].append({"arm": arm, "scene": scene, "attempt": n,
                                       "dir": os.path.basename(dst)})
    # remove from done + clear the contamination flag -> pair re-enters queue
    s["done"] = [d for d in s["done"]
                 if not (d["arm"] == arm and d["scene"] == scene)]
    retried.append(key)

for key in retried:
    s["contaminated"].pop(key, None)

json.dump(s, open(STATE, "w"), indent=2)
print("retried:", retried)
print("done now:", [(d["arm"], d["scene"]) for d in s["done"]])
EOF

echo "== relaunch scheduler =="
setsid nohup bash /mnt/storage_pool/liaoyuanjun/launch_scheduler.sh < /dev/null > /dev/null 2>&1 &
sleep 5
N=$(ps aux | grep 'final30k_[s]cheduler.py' | grep -v grep | wc -l)
echo "schedulers after relaunch: $N"
tail -2 /mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler.log
