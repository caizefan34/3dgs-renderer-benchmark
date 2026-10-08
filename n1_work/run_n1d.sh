#!/bin/bash
# N1D execution: freeze (identity-gated) -> actions+timing (GPU0/GPU1) ->
# finalize (raw artifacts only; no analysis, no verdict).
set -e
cd /root/epic05_n1d
PY=/root/miniforge3/envs/gsplat/bin/python
mkdir -p artifacts logs

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader > cont_gpu_start.csv

$PY -m py_compile n1d_freeze_partners.py n1d_pass_actions.py n1d_timing.py \
  n1d_finalize.py
echo "syntax OK"

echo "=== N1D freeze + identity verification (before any action) ==="
$PY n1d_freeze_partners.py 2>&1 | tee logs/freeze.log

run_scene () {
  local scene=$1 gpu=$2
  {
    echo "=== [$scene] n1d actions (GPU$gpu) ==="
    CUDA_VISIBLE_DEVICES=$gpu $PY n1d_pass_actions.py $scene || { echo "ACTIONS FAIL $scene"; exit 1; }
    echo "=== [$scene] n1d timing (GPU$gpu) ==="
    CUDA_VISIBLE_DEVICES=$gpu $PY n1d_timing.py $scene || { echo "TIMING FAIL $scene"; exit 1; }
    echo "=== [$scene] N1D PIPELINE COMPLETE ==="
  } > logs/n1d_$scene.log 2>&1
}

run_scene room 0 &
P0=$!
run_scene bicycle 1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "room rc=$R0 bicycle rc=$R1"
echo "=== room tail ==="; tail -5 logs/n1d_room.log
echo "=== bicycle tail ==="; tail -5 logs/n1d_bicycle.log
[ $R0 -eq 0 ] && [ $R1 -eq 0 ] || exit 1

echo "=== N1D finalize ==="
$PY n1d_finalize.py
