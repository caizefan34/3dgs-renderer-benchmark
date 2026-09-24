#!/bin/bash
# Watch scheduler stability streak (read-only).
for i in 1 2 3 4 5; do
  date +%H:%M:%S
  python3 - <<'EOF'
import json
s = json.load(open("/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"))
print(s["clean_seen_at"], "active:", len(s["active"]), "launched:", s["launched_any"])
EOF
  sleep 55
done
