#!/bin/bash
# WV0 campaign driver: run all 9 states across GPUs 4-7.
export WV0_THREADS=8
PY=/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python
RUN=/dev/shm/_wv0_run.py
CAMPAIGN=/dev/shm/_wv0_campaign_state.py
OUTDIR=/dev/shm/wv0_out/states
mkdir -p $OUTDIR
LOGDIR=/dev/shm/wv0_out/logs
mkdir -p $LOGDIR

run_state () {
  local sid=$1 gpu=$2
  local out=$OUTDIR/${sid}.pt
  local log=$LOGDIR/${sid}.log
  if [ -f "$out" ]; then echo "SKIP $sid (exists)"; return; fi
  echo "START $sid on gpu $gpu $(date)"
  $PY $RUN $CAMPAIGN $gpu $OUTDIR/${sid}.json $sid 540p 8 $out 0 > $log 2>&1
  echo "DONE $sid rc=$? $(date)"
}

run_state room__iter5000 4 &
run_state room__iter15000 5 &
run_state room__iter30000 6 &
run_state garden__iter5000 7 &
wait
run_state garden__iter15000 4 &
run_state garden__iter30000 5 &
run_state bicycle__iter5000 6 &
run_state bicycle__iter15000 7 &
wait
run_state bicycle__iter30000 4 &
wait
echo "ALL DONE $(date)"
