#!/bin/bash
cd /root/reprotopo0_src
PY=/root/miniforge3/envs/gsplat/bin/python
DET=/root/reprotopo0_s0/_det
mkdir -p $DET
run() {
  local snap=$1 out=$2 base=$3
  PYTORCH_ALLOC_CONF=expandable_segments:True $PY rt0_determinism.py --snap "$snap" --out "$out" --gpu_base "$base" > "$out.log" 2>&1
  echo "[$(date +%T)] done $out exit=$?"
}
CAP=/root/reprotopo0_s0/cap
run $CAP/room/event_10200.pt $DET/room_10200.json 0 &
run $CAP/room/event_10300.pt $DET/room_10300.json 4 &
wait
run $CAP/room/event_10400.pt $DET/room_10400.json 0 &
run $CAP/room/event_10500.pt $DET/room_10500.json 4 &
wait
run $CAP/bicycle/event_10100.pt $DET/bicycle_10100.json 0 &
run $CAP/bicycle/event_10200.pt $DET/bicycle_10200.json 4 &
wait
run $CAP/bicycle/event_10300.pt $DET/bicycle_10300.json 0 &
run $CAP/bicycle/event_10400.pt $DET/bicycle_10400.json 4 &
wait
run $CAP/bicycle/event_10500.pt $DET/bicycle_10500.json 0 &
wait
echo "ALL_DET_DONE"
