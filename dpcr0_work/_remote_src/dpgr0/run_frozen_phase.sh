#!/usr/bin/env bash
# DPGR0 frozen-phase driver: run all 4 cells sequentially, each on a free GPU.
set -uo pipefail
cd /root/epic05_dpgr0
PY=/root/miniforge3/envs/gsplat/bin/python
CELLS=("room@10000" "room@15000" "bicycle@10000" "bicycle@15000")
mkdir -p runs/dpgr0_phase logs

pick_gpu() {
  while true; do
    while read -r idx util memused memtot; do
      util_n=${util%% *}
      memused_n=${memused%% *}
      if [ "$util_n" -lt 40 ] && [ "$memused_n" -lt 20000 ]; then
        echo "$idx"; return
      fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | sed 's/,//g')
    sleep 20
  done
}

for cell in "${CELLS[@]}"; do
  gpu=$(pick_gpu)
  log="logs/phase_${cell}_gpu${gpu}_frozen.log"
  echo "[driver] $(date -u +%H:%M:%S) launching cell=$cell on gpu=$gpu" | tee -a logs/driver.log
  $PY dpgr0_phase.py "$gpu" "$cell" 128 32 > "$log" 2>&1
  rc=$?
  echo "[driver] $(date -u +%H:%M:%S) cell=$cell gpu=$gpu rc=$rc" | tee -a logs/driver.log
  if [ $rc -ne 0 ]; then
    echo "[driver] ERROR cell=$cell rc=$rc see $log" | tee -a logs/driver.log
  fi
done
echo "[driver] ALL DONE" | tee -a logs/driver.log