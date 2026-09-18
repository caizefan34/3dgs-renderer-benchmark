#!/bin/bash
# R4 Auto-Launch: Wait for GPUs to free up, then launch 13-scene training
#
# Monitors GPU memory. When at least 1 GPU has < 5GB used, launches training
# on available GPUs. Runs in background with nohup.
#
# Usage: nohup bash experiments/r4/auto_launch.sh &

set -uo pipefail
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"

REPO="$HOME/3dgs-renderer-benchmark"
LOG="$HOME/r4_auto_launch.log"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2"
LOG_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs"

mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

echo "[$(date)] R4 Auto-Launch monitoring started" | tee -a "$LOG"

# Wait for GPUs to free up (at least 8 GPUs with < 5GB used)
while true; do
  # Count GPUs with < 5GB used
  FREE_GPUS=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 < 5000 {count++} END {print count+0}')
  
  if [ "$FREE_GPUS" -ge 8 ]; then
    echo "[$(date)] All 8 GPUs free! Launching training." | tee -a "$LOG"
    break
  elif [ "$FREE_GPUS" -ge 1 ]; then
    echo "[$(date)] $FREE_GPUS GPU(s) free, need 8. Waiting..." | tee -a "$LOG"
  else
    echo "[$(date)] No GPUs free. Waiting..." | tee -a "$LOG"
  fi
  
  sleep 60  # Check every minute
done

# Launch the 13-scene training
echo "[$(date)] Starting 13-scene training..." | tee -a "$LOG"
bash "$REPO/experiments/r4/run_r4_13scene_v2.sh" 2>&1 | tee -a "$LOG"

echo "[$(date)] Training complete." | tee -a "$LOG"
