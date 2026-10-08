#!/usr/bin/env bash
# Robust parallel frozen-phase launcher: each cell on a verified-free GPU.
set -uo pipefail
cd /root/epic05_dpgr0
PY=/root/miniforge3/envs/gsplat/bin/python
mkdir -p runs/dpgr0_phase logs

# gpu_uuid <index>
gpu_uuid() { nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1"; }
# count foreign compute-apps on a gpu index (excluding our own pids)
foreign_count() {
  local idx="$1"; local uu; uu=$(gpu_uuid "$idx")
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F", " -v u="$uu" '$1==u{c++} END{print c+0}'
}
# wait until gpu has zero compute-apps, then claim
wait_free() {
  local idx="$1"
  while true; do
    fc=$(foreign_count "$idx")
    if [ "$fc" -eq 0 ]; then echo "$idx"; return; fi
    sleep 15
  done
}

launch() {
  local want="$1"; local cell="$2"
  local gpu; gpu=$(wait_free "$want")
  local log="logs/phase_${cell}_gpu${gpu}_frozen.log"
  echo "[launch] $(date -u +%H:%M:%S) cell=$cell gpu=$gpu (was free)"
  nohup "$PY" dpgr0_phase.py "$gpu" "$cell" 128 32 > "$log" 2>&1 &
  echo "[launch] cell=$cell gpu=$gpu pid=$! log=$log"
}

launch 4 "room@10000"
launch 5 "room@15000"
launch 6 "bicycle@10000"
launch 7 "bicycle@15000"
echo "[launch] all launched"