#!/usr/bin/env bash
set -uo pipefail
cd /root/epic05_dpgr0
PY=/root/miniforge3/envs/gsplat/bin/python
CELLS=("room@10000" "room@15000" "bicycle@10000" "bicycle@15000")
GPUS=(4 5 6 7)
mkdir -p runs/dpgr0_phase logs
for i in "${!CELLS[@]}"; do
  cell="${CELLS[$i]}"; gpu="${GPUS[$i]}"
  log="logs/phase_${cell}_gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY dpgr0_phase.py $gpu "$cell" 128 32 > "$log" 2>&1 &
  echo "launched cell=$cell gpu=$gpu pid=$! log=$log"
done
echo "all launched"
