#!/bin/bash
cd /root/reprotopo0_src
PY=/root/miniforge3/envs/gsplat/bin/python
T=/root/reprotopo0_s0/_timing
mkdir -p $T
run() {
  local snap=$1 out=$2 gpu=$3
  PYTORCH_ALLOC_CONF=expandable_segments:True $PY rt0_timing.py --snap "$snap" --out "$out" --gpu_base "$gpu" --repeats 256 --n_per_iter 12 > "$out.log" 2>&1
  echo "[$(date +%T)] done $out exit=$?"
}
CAP=/root/reprotopo0_s0/cap
run $CAP/bicycle/event_10100.pt $T/bicycle_10100.json 0 &
run $CAP/bicycle/event_10200.pt $T/bicycle_10200.json 1 &
run $CAP/bicycle/event_10300.pt $T/bicycle_10300.json 2 &
run $CAP/bicycle/event_10400.pt $T/bicycle_10400.json 3 &
run $CAP/bicycle/event_10500.pt $T/bicycle_10500.json 4 &
wait
run $CAP/room/event_10100.pt $T/room_10100.json 5 &
run $CAP/room/event_10200.pt $T/room_10200.json 6 &
run $CAP/room/event_10300.pt $T/room_10300.json 7 &
wait
run $CAP/room/event_10400.pt $T/room_10400.json 5 &
run $CAP/room/event_10500.pt $T/room_10500.json 6 &
wait
echo "ALL_TIMING_DONE"
