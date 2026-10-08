#!/bin/bash
# Room order-swapped re-pair: VOS arm FIRST, then C0 — disambiguates
# thermal/clock drift from real stage savings (first run of a pair tends to
# run on a cooler GPU; the original pair ran C0 first and showed VOS 1.2%
# slower on IDENTICAL pre-region work).
set -u
export PY=/root/miniforge3/envs/gsplat/bin/python
export D=/root/vos_traces
export R=$D/vos_runs
cd "$D"
(
  echo "[$(date -u +%FT%TZ)] room-repair VOS arm first (GPU 3)"
  CUDA_VISIBLE_DEVICES=3 $PY vos_train_driver.py --scene room \
    --iterations 30000 --gpu 3 --vos-start 15000 --final-eval all \
    --outdir "$R"/vos_room_30k_b > "$D"/logs/vos_vos_room_30k_b.log 2>&1
  echo "[$(date -u +%FT%TZ)] room-repair VOS done (exit $?)"
  echo "[$(date -u +%FT%TZ)] room-repair C0 arm second (GPU 3)"
  CUDA_VISIBLE_DEVICES=3 $PY vos_train_driver.py --scene room \
    --iterations 30000 --gpu 3 --vos-start 99999 --final-eval all \
    --outdir "$R"/c0_room_30k_b > "$D"/logs/vos_c0_room_30k_b.log 2>&1
  echo "[$(date -u +%FT%TZ)] room-repair C0 done (exit $?)"
) > "$D"/logs/scene_room_30k_repair.log 2>&1 &
echo "ROOM-REPAIR pid=$! LAUNCHED $(date -u +%FT%TZ)"
