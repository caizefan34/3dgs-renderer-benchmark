#!/bin/bash
# VOS Phase-4 launch: 3 traced 30K runs + 6 checkpoint runs (5K/15K) on EPIC-05.
# Trace runs (T30): room GPU0, bicycle GPU1, garden GPU3.
# Ckpt runs: room_R15 GPU4, bicycle_R15 GPU5, garden_R15 GPU6 (+garden_R5 after),
#            room_R5 GPU7 (+bicycle_R5 after).
# All runs are FUNCTIONAL_ONLY (timing not used for gate claims; stage timings come
# from the P12/P16 measured campaign + the isolated microbench).
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
T=/root/epic05_p13x/publication_trainer_p13x.py
mkdir -p "$D"/logs
cd "$D"

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader \
  > "$D"/logs/gpu_before.csv
date -u +%FT%TZ >> "$D"/logs/gpu_before.csv

TRAINER_ARGS="--arm c0 --loss fused --eps2d 0.3 --final-eval all --timing-grade FUNCTIONAL_ONLY --lpips"

# ---- traced 30K runs (driver) ----
setsid nohup $PY "$D"/vos_trace_driver.py --scene room --iterations 30000 --gpu 0 \
  --final-eval all --outdir "$D"/room_T30 \
  < /dev/null > "$D"/logs/room_T30.log 2>&1 &
echo "room_T30 pid=$!"
setsid nohup $PY "$D"/vos_trace_driver.py --scene bicycle --iterations 30000 --gpu 1 \
  --final-eval all --outdir "$D"/bicycle_T30 \
  < /dev/null > "$D"/logs/bicycle_T30.log 2>&1 &
echo "bicycle_T30 pid=$!"
setsid nohup $PY "$D"/vos_trace_driver.py --scene garden --iterations 30000 --gpu 3 \
  --final-eval all --outdir "$D"/garden_T30 \
  < /dev/null > "$D"/logs/garden_T30.log 2>&1 &
echo "garden_T30 pid=$!"

# ---- 15K checkpoint runs (plain trainer, no trace) ----
setsid nohup $PY "$T" $TRAINER_ARGS --scene room --iterations 15000 --gpu 4 \
  --outdir "$D"/room_R15 \
  < /dev/null > "$D"/logs/room_R15.log 2>&1 &
echo "room_R15 pid=$!"
setsid nohup $PY "$T" $TRAINER_ARGS --scene bicycle --iterations 15000 --gpu 5 \
  --outdir "$D"/bicycle_R15 \
  < /dev/null > "$D"/logs/bicycle_R15.log 2>&1 &
echo "bicycle_R15 pid=$!"
setsid nohup bash -c "
  $PY '$T' $TRAINER_ARGS --scene garden --iterations 15000 --gpu 6 --outdir '$D'/garden_R15 \
    > '$D'/logs/garden_R15.log 2>&1
  echo GARDEN_R15_EXIT=\$? >> '$D'/logs/chain_gpu6.log
  $PY '$T' $TRAINER_ARGS --scene garden --iterations 5000 --gpu 6 --outdir '$D'/garden_R5 \
    > '$D'/logs/garden_R5.log 2>&1
  echo GARDEN_R5_EXIT=\$? >> '$D'/logs/chain_gpu6.log
" < /dev/null > /dev/null 2>&1 &
echo "gpu6 chain (garden_R15 -> garden_R5) pid=$!"

# ---- 5K checkpoint runs (chained on gpu 7) ----
setsid nohup bash -c "
  $PY '$T' $TRAINER_ARGS --scene room --iterations 5000 --gpu 7 --outdir '$D'/room_R5 \
    > '$D'/logs/room_R5.log 2>&1
  echo ROOM_R5_EXIT=\$? >> '$D'/logs/chain_gpu7.log
  $PY '$T' $TRAINER_ARGS --scene bicycle --iterations 5000 --gpu 7 --outdir '$D'/bicycle_R5 \
    > '$D'/logs/bicycle_R5.log 2>&1
  echo BICYCLE_R5_EXIT=\$? >> '$D'/logs/chain_gpu7.log
" < /dev/null > /dev/null 2>&1 &
echo "gpu7 chain (room_R5 -> bicycle_R5) pid=$!"

echo "ALL_LAUNCHED $(date -u +%FT%TZ)"
