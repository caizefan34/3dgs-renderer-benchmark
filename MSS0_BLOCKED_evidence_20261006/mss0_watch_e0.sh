#!/bin/bash
# MSS0 E0 unattended auto-launch watcher (CPU-only E0; guarded launcher enforces all gates).
# Polls allowed GPUs {1,2,3,4}; when one is idle (util<=2%, mem<=512MiB, no compute process)
# for STREAK consecutive polls, launches E0 via scripts/run_on_mx.py. Retries on transient
# launcher failure; exits once E0 launcher returns 0 (or on timeout / prior-result idempotency).
set -u
PKG=/mnt/storage_pool/liaoyuanjun/mss0_mx/MSS0_MX_DSH
WORK=/mnt/storage_pool/liaoyuanjun/mss0_mx
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
ALLOWED=(1 2 3 4)
IDLE_UTIL=2
IDLE_MEM=512
POLL=60
STREAK=2
MAX_HOURS=12
LOG=${WORK}/watch_e0.log

echo "$(date -u +%FT%TZ) watcher start; allowed=${ALLOWED[*]} poll=${POLL}s streak=${STREAK}" >> "$LOG"

if ls ${WORK}/*/e0/E0.json >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) prior E0 result present; watcher exits" >> "$LOG"
  exit 0
fi

deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
declare -A idle_count
while [ "$(date +%s)" -lt "$deadline" ]; do
  snap=$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null)
  launched=0
  for idx in "${ALLOWED[@]}"; do
    line=$(echo "$snap" | awk -F, -v i="$idx" 'gsub(/ /,""); if($1==i){print}')
    [ -z "$line" ] && continue
    uuid=$(echo "$line" | awk -F, '{print $2}' | tr -d ' ')
    util=$(echo "$line" | awk -F, '{print $3}' | tr -d ' ')
    mem=$(echo "$line" | awk -F, '{print $4}' | tr -d ' ')
    busy=$(echo "$apps" | grep -c "$uuid")
    if [ "${util:-100}" -le "$IDLE_UTIL" ] && [ "${mem:-99999}" -le "$IDLE_MEM" ] && [ "${busy:-1}" -eq 0 ]; then
      idle_count[$idx]=$(( ${idle_count[$idx]:-0} + 1 ))
      if [ "${idle_count[$idx]}" -ge "$STREAK" ]; then
        RUNID="MSS0_$(date -u +%Y%m%d_%H%M%S)_e0auto"
        echo "$(date -u +%FT%TZ) LAUNCH E0 run=$RUNID gpu=$uuid (streak ${idle_count[$idx]})" >> "$LOG"
        cd "$PKG" || { echo "cd PKG failed" >> "$LOG"; sleep "$POLL"; continue; }
        "$PY" scripts/run_on_mx.py --run-id "$RUNID" --gpu-uuid "$uuid" --stage e0 --entry scripts/run_e0.py --max-minutes 20 >> "$LOG" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
          echo "$(date -u +%FT%TZ) E0 OK run=$RUNID ; stage=${WORK}/${RUNID}/e0" >> "$LOG"
          launched=1
          break
        else
          echo "$(date -u +%FT%TZ) E0 launcher rc=$rc (transient); reset streak, keep polling" >> "$LOG"
          idle_count[$idx]=0
        fi
      fi
    else
      idle_count[$idx]=0
    fi
  done
  [ "$launched" -eq 1 ] && break
  sleep "$POLL"
done
echo "$(date -u +%FT%TZ) watcher done" >> "$LOG"
