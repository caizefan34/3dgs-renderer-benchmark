#!/bin/bash
# Requeue b0_counter after contamination (c0e01_room precedent):
# 1. stop scheduler (in-flight runs keep running; restarted scheduler adopts by pid)
# 2. rename contaminated run dir with _contaminated_attempt1 suffix (forensics kept)
# 3. drop the b0_counter record from state done list
# 4. relaunch scheduler; queue entry re-fires on a clean GPU
set -e
SCHED_PIDS=$(pgrep -f 'publication_scheduler.py')
echo "scheduler pids: $SCHED_PIDS"
if [ -n "$SCHED_PIDS" ]; then
  kill $SCHED_PIDS
  sleep 3
  pgrep -f 'publication_scheduler.py' && { echo "STILL ALIVE"; exit 1; } || echo "scheduler stopped"
fi
cd /mnt/storage_pool/liaoyuanjun/pub_runs
if [ -d b0_counter ] && [ ! -d b0_counter_contaminated_attempt1 ]; then
  mv b0_counter b0_counter_contaminated_attempt1
  echo "renamed b0_counter -> b0_counter_contaminated_attempt1"
fi
python3 - << 'EOF'
import json
SP = "/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"
st = json.load(open(SP))
before = len(st["done"])
st["done"] = [d for d in st["done"] if d["rid"] != "b0_counter"]
st.pop("contaminated", {}).pop("b0_counter", None) if isinstance(st.get("contaminated"), dict) else None
json.dump(st, open(SP, "w"), indent=1)
print(f"state done list: {before} -> {len(st['done'])} (b0_counter removed)")
EOF
cd /mnt/storage_pool/liaoyuanjun
setsid nohup /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python /mnt/storage_pool/liaoyuanjun/publication_scheduler.py >> /mnt/storage_pool/liaoyuanjun/pub_scheduler_stdout.log 2>&1 < /dev/null &
sleep 8
pgrep -f 'publication_scheduler.py' && echo "scheduler relaunched" || { echo "REL launch FAILED"; exit 1; }
tail -8 /mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler.log
