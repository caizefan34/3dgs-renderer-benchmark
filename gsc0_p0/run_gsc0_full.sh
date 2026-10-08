#!/bin/bash
# GSC0 P0 full trace collection: room/garden/bicycle @ 30K, phases {1000,15000,30000}
PY=/root/miniforge3/envs/gsplat/bin/python
OUT=/root/gsc0_run
mkdir -p "$OUT"
cd /root
echo "launch time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for pair in "room 3" "garden 4" "bicycle 5"; do
  set -- $pair
  scene=$1; gpu=$2
  nohup "$PY" /root/gsc0_trace.py --scene "$scene" --gpu "$gpu" \
      --out "$OUT/$scene" --iterations 30000 \
      > "$OUT/$scene.log" 2>&1 &
  echo "launched $scene on GPU $gpu pid $!"
done
echo "all launched"