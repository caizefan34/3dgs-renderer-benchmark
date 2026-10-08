#!/bin/bash
cd /root/reprotopo0_src
PY=/root/miniforge3/envs/gsplat/bin/python
CAP=/root/reprotopo0_s0/cap
PAR=/root/reprotopo0_s0/_parity
DET=/root/reprotopo0_s0/_det
mkdir -p $PAR $DET
export PYTORCH_ALLOC_CONF=expandable_segments:True

# parity for the 8 remaining events (fast, one per GPU 0-7)
$PY rt0_parity.py --snap $CAP/room/event_10200.pt    --out $PAR/room_10200.json    --device cuda:0 > $PAR/room_10200.log 2>&1 &
$PY rt0_parity.py --snap $CAP/room/event_10300.pt    --out $PAR/room_10300.json    --device cuda:1 > $PAR/room_10300.log 2>&1 &
$PY rt0_parity.py --snap $CAP/room/event_10400.pt    --out $PAR/room_10400.json    --device cuda:2 > $PAR/room_10400.log 2>&1 &
$PY rt0_parity.py --snap $CAP/room/event_10500.pt    --out $PAR/room_10500.json    --device cuda:3 > $PAR/room_10500.log 2>&1 &
$PY rt0_parity.py --snap $CAP/bicycle/event_10200.pt --out $PAR/bicycle_10200.json --device cuda:4 > $PAR/bicycle_10200.log 2>&1 &
$PY rt0_parity.py --snap $CAP/bicycle/event_10300.pt --out $PAR/bicycle_10300.json --device cuda:5 > $PAR/bicycle_10300.log 2>&1 &
$PY rt0_parity.py --snap $CAP/bicycle/event_10400.pt --out $PAR/bicycle_10400.json --device cuda:6 > $PAR/bicycle_10400.log 2>&1 &
$PY rt0_parity.py --snap $CAP/bicycle/event_10500.pt --out $PAR/bicycle_10500.json --device cuda:7 > $PAR/bicycle_10500.log 2>&1 &
wait
echo "PARITY_REM_DONE"

det() { $PY rt0_determinism.py --snap "$1" --out "$2" --gpu_base "$3" > "$2.log" 2>&1; echo "[$(date +%T)] done $2 exit=$?"; }
# determinism: 2 concurrent (each needs 4 GPUs), remaining 8 events
det $CAP/room/event_10200.pt    $DET/room_10200.json    0 &
det $CAP/bicycle/event_10200.pt $DET/bicycle_10200.json 4 &
wait
det $CAP/room/event_10300.pt    $DET/room_10300.json    0 &
det $CAP/bicycle/event_10300.pt $DET/bicycle_10300.json 4 &
wait
det $CAP/room/event_10400.pt    $DET/room_10400.json    0 &
det $CAP/bicycle/event_10400.pt $DET/bicycle_10400.json 4 &
wait
det $CAP/room/event_10500.pt    $DET/room_10500.json    0 &
det $CAP/bicycle/event_10500.pt $DET/bicycle_10500.json 4 &
wait
echo "ALL_PAR_DET_DONE"
