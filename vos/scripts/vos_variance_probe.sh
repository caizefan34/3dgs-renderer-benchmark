#!/bin/bash
# Decisive checks on the VOS-arm variance anomaly:
# 1. bitmaps_captured vs iterations (1 render/iter expected)
# 2. visibility evolution in the logs (VOS_a vs VOS_b vs C0)
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
$PY - <<'EOF'
import json
for d in ["vos_room_30k", "vos_room_30k_b", "c0_room_30k",
          "vos_bicycle_30k", "vos_garden_30k"]:
    try:
        st = json.load(open(f"vos_runs/{d}/vos_stats.json"))
    except FileNotFoundError:
        continue
    print(f"{d}: bitmaps={st.get('bitmaps_captured')} iters="
          f"{st.get('iterations')} ratio="
          f"{st.get('bitmaps_captured')}/{st.get('iterations')} "
          f"active_sum={st.get('n_active_rows_sum')} "
          f"steps={st.get('n_vos_steps')}")
EOF
echo "=== vis evolution (log greps) ==="
for d in vos_vos_room_30k vos_vos_room_30k_b vos_c0_room_30k vos_c0_room_30k_b; do
  echo "-- $d:"
  grep -oE "iter (16000|20000|25000|30000)\].*vis=[0-9,]+" "/root/vos_traces/logs/${d}.log" 2>/dev/null \
    | sed -E 's/loss=[0-9.]+ //;s/GS=[0-9,]+ //' | head -4
done
