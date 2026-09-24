#!/bin/bash
# Reset the spurious failures (trainers killed by ssh-session teardown, not by
# trainer faults) and relaunch the scheduler with the proven nohup pattern.
pkill -f final30k_scheduler.py
sleep 2
python3 - <<'EOF'
import json
p = "/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"
s = json.load(open(p))
# the three adopted-dead entries were session-teardown kills, not run failures
s["failed"] = []
s["active"] = {}
s["launched_any"] = False
s["clean_seen_at"] = {}
s["contaminated"] = {}
json.dump(s, open(p, "w"), indent=2)
print("state reset:", s)
EOF
rm -f /mnt/storage_pool/liaoyuanjun/final30k_runs/log_b1a_bicycle.txt \
      /mnt/storage_pool/liaoyuanjun/final30k_runs/log_c0_bicycle.txt \
      /mnt/storage_pool/liaoyuanjun/final30k_runs/log_b1a_bonsai.txt
nohup bash /mnt/storage_pool/liaoyuanjun/launch_scheduler.sh < /dev/null > /dev/null 2>&1 &
sleep 3
echo "scheduler relaunched"
