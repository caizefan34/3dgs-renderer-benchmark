#!/usr/bin/env bash
# Frozen-phase launcher: one cell per distinct GPU (patched det32).
set -uo pipefail
cd /root/epic05_dpgr0
PY=/root/miniforge3/envs/gsplat/bin/python
mkdir -p runs/dpgr0_phase logs
launch() {
  local gpu="$1"; local cell="$2"
  local log="logs/phase_${cell}_gpu${gpu}_frozen.log"
  echo "[launch] $(date -u +%H:%M:%S) cell=$cell gpu=$gpu"
  nohup "$PY" dpgr0_phase.py "$gpu" "$cell" 128 32 > "$log" 2>&1 &
  echo "[launch] cell=$cell gpu=$gpu pid=$! log=$log"
}
launch 2 "room@10000"
launch 3 "room@15000"
launch 4 "bicycle@10000"
launch 5 "bicycle@15000"
echo "[launch] all launched"