#!/bin/bash
# R4 Auto-Launch v2: Wait for GPUs to free up, then launch 13-scene training
set -uo pipefail
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"

REPO="$HOME/3dgs-renderer-benchmark"
LOG="$HOME/r4_auto_launch_v2.log"

echo "[$(date)] R4 Auto-Launch v2 monitoring started" | tee -a "$LOG"

while true; do
  FREE_GPUS=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 < 5000 {count++} END {print count+0}')
  
  if [ "$FREE_GPUS" -ge 8 ]; then
    echo "[$(date)] All 8 GPUs free! Launching 13-scene training." | tee -a "$LOG"
    break
  fi
  
  echo "[$(date)] $FREE_GPUS GPU(s) free, need 8. Waiting..." | tee -a "$LOG"
  sleep 120
done

bash "$REPO/experiments/r4/run_r4_13scene_v3.sh" 2>&1 | tee -a "$LOG"
echo "[$(date)] Training complete." | tee -a "$LOG"
