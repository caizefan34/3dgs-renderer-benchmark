#!/bin/bash
# Fix-2 rollout: kill scheduler, inject the live orphan c0/bonsai into state,
# relaunch the fixed scheduler (active-pair exclusion + adoption).
set -e
pkill -f 'final30k_scheduler.py' || true
sleep 3
pkill -f 'final30k_scheduler.py' || true   # second sweep for slow exits
sleep 1
python3 - <<'EOF'
import json
p = "/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"
s = json.load(open(p))
# Inject the live orphan launched 19:17:33 by the previous incarnation.
# pid verified alive (99.9% CPU, GPU 0). started ~= epoch of 19:17:33 (+0800).
s["active"] = {
    "0": {"arm": "c0", "scene": "bonsai", "pid": 2123394,
          "started": 1790161053, "timing_grade": "PUBLICATION"}
}
s["launched_any"] = True
s["clean_seen_at"] = {}
s["contaminated"] = {}
s["done"] = []
s["failed"] = []
json.dump(s, open(p, "w"), indent=2)
print("state injected:", s["active"])
EOF
nohup bash /mnt/storage_pool/liaoyuanjun/launch_scheduler.sh < /dev/null > /dev/null 2>&1 &
sleep 4
echo "relaunched"
