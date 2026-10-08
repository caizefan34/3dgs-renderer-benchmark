#!/bin/bash
cd /root/reprotopo0_src
PY=/root/miniforge3/envs/gsplat/bin/python
BASE=/root/reprotopo0_s0/_baseline
mkdir -p $BASE
run() {
  local snap=$1 out=$2 base=$3
  PYTORCH_ALLOC_CONF=expandable_segments:True $PY rt0_baseline.py --snap "$snap" --out "$out" --gpu_base "$base" > "$out.log" 2>&1
  echo "[$(date +%T)] done $out exit=$?"
}
CAP=/root/reprotopo0_s0/cap
run $CAP/room/event_10100.pt $BASE/room_10100.json 0 &
run $CAP/room/event_10200.pt $BASE/room_10200.json 4 &
wait
run $CAP/room/event_10300.pt $BASE/room_10300.json 0 &
run $CAP/room/event_10400.pt $BASE/room_10400.json 4 &
wait
run $CAP/room/event_10500.pt $BASE/room_10500.json 0 &
run $CAP/bicycle/event_10100.pt $BASE/bicycle_10100.json 4 &
wait
run $CAP/bicycle/event_10200.pt $BASE/bicycle_10200.json 0 &
run $CAP/bicycle/event_10300.pt $BASE/bicycle_10300.json 4 &
wait
run $CAP/bicycle/event_10400.pt $BASE/bicycle_10400.json 0 &
run $CAP/bicycle/event_10500.pt $BASE/bicycle_10500.json 4 &
wait
echo "ALL_BASE_DONE"
