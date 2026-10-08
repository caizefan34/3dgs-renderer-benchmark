#!/bin/bash
# Re-run DSH-7/8/9 Phase A for garden + bicycle to capture the full 7-checkpoint
# schedule (the first pass was terminated before the 25000 capture). Deterministic
# repro: same config, seed, preloading. Targets GPUs 6 (garden) and 7 (bicycle).
set -u
W=/root/dsh_20261001
PY=/root/miniforge3/envs/p16_fastergs/bin/python
ITERS=500,2000,5000,10000,14500,20000,25000
for scene in garden bicycle; do
  case $scene in
    garden) gpu=6;;
    bicycle) gpu=7;;
  esac
  trace=$W/runs/DSH_PHASEA/$scene/trace
  # clear stale partial jsonl so the fresh run writes a clean 7-record file;
  # keep raw npz (will be overwritten identically by the deterministic repro)
  if [ -f $trace/phase_a_records.jsonl ]; then
    mv $trace/phase_a_records.jsonl $trace/phase_a_records.jsonl.bak
  fi
  rm -f $trace/trace_meta.json
  log=$W/runs/DSH_PHASEA/$scene/train_rerun.log
  CUDA_VISIBLE_DEVICES=$gpu DSH_TRACE=1 DSH_TRACE_OUT=$trace \
    DSH_TRACE_TAG=PHASEA_$scene DSH_TRACE_SEED=42 DSH_TRACE_ITERS=$ITERS \
    setsid nohup $PY $W/instr/nerficg/scripts/train.py \
    -c $W/configs/dsh789_full_$scene.yaml > $log 2>&1 < /dev/null &
  echo "launched $scene on GPU $gpu pid $! log=$log"
done
