#!/bin/bash
# P15-P0 Step 3 driver: six cells, ONE clean GPU queue (sequential).
# Any export that fails the P14 reproduction gate stops the queue with
# P0_EXPORT_INVALID.
set -u
ROOT=/root/epic05_p15/p0
P=/root/miniforge3/envs/gsplat/bin/python
GPU=${P15_GPU:-4}
mkdir -p "$ROOT/exports" "$ROOT/summaries" "$ROOT/logs"
cd "$ROOT/scripts"
rc_all=0
for spec in "room 0" "room 1" "room 2" "bicycle 0" "bicycle 1" "bicycle 2"; do
  set -- $spec; SC=$1; CE=$2
  LOG="$ROOT/logs/export_${SC}_cell${CE}.log"
  echo "=== export ${SC} cell ${CE} on GPU ${GPU} ==="
  CUDA_VISIBLE_DEVICES=$GPU $P p15_p0_export.py \
    --p14-scripts /root/epic05_p14/scripts \
    --p14-root /root/epic05_p14 \
    --p15-root "$ROOT" \
    --scene "$SC" --cell "$CE" --gpu "$GPU" \
    --out "$ROOT/exports/p15_export_${SC}_cell${CE}.npz" \
    --summary "$ROOT/summaries/p15_export_${SC}_cell${CE}.json" \
    > "$LOG" 2>&1
  rc=$?
  tail -n 14 "$LOG"
  if [ $rc -ne 0 ]; then
    echo "EXPORT FAILED rc=$rc -> P0_EXPORT_INVALID (queue stopped)"
    rc_all=$rc
    break
  fi
done
echo "export queue done rc=$rc_all"
exit $rc_all
