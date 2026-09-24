#!/bin/bash
# Requeue b1as43_train after contamination (same procedure as b0_counter).
# Scheduler already exited (ALL_RUNS_COMPLETE) - no process to stop.
set -e
pgrep -f 'publication_scheduler.py' && { echo "scheduler still running - abort"; exit 1; } || echo "scheduler not running (ok)"
cd /mnt/storage_pool/liaoyuanjun/pub_runs
if [ -d b1as43_train ] && [ ! -d b1as43_train_contaminated_attempt1 ]; then
  mv b1as43_train b1as43_train_contaminated_attempt1
  echo "renamed b1as43_train -> b1as43_train_contaminated_attempt1"
fi
python3 - << 'EOF'
import json
SP = "/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"
st = json.load(open(SP))
before = len(st["done"])
st["done"] = [d for d in st["done"] if d["rid"] != "b1as43_train"]
if isinstance(st.get("contaminated"), dict):
    st["contaminated"].pop("b1as43_train", None)
json.dump(st, open(SP, "w"), indent=1)
print(f"state done list: {before} -> {len(st['done'])}")
EOF
cd /mnt/storage_pool/liaoyuanjun
setsid nohup /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python /mnt/storage_pool/liaoyuanjun/publication_scheduler.py >> /mnt/storage_pool/liaoyuanjun/pub_scheduler_stdout.log 2>&1 < /dev/null &
sleep 8
pgrep -f 'publication_scheduler.py' && echo "scheduler relaunched" || { echo "RELAUNCH FAILED"; exit 1; }
tail -5 /mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler.log
