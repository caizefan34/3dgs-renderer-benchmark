#!/bin/bash
# N1 Stage 2 (grads/conflict/sketches + counterfactual actions) and Stage 3
# (timing), room on GPU0 and bicycle on GPU1, per-scene pipeline sequential.
cd /root/epic05_n1
PY=/root/miniforge3/envs/gsplat/bin/python
mkdir -p logs artifacts

$PY -m py_compile n1_core.py n1_pass_grads.py n1_pass_actions.py n1_stage3_timing.py \
  || { echo "SYNTAX FAIL"; exit 1; }
echo "syntax OK"

run_scene () {
  local scene=$1 gpu=$2
  {
    echo "=== [$scene] pass_grads (GPU$gpu) ==="
    CUDA_VISIBLE_DEVICES=$gpu $PY n1_pass_grads.py $scene || { echo "GRADS FAIL $scene"; exit 1; }
    echo "=== [$scene] pass_actions (GPU$gpu) ==="
    CUDA_VISIBLE_DEVICES=$gpu $PY n1_pass_actions.py $scene || { echo "ACTIONS FAIL $scene"; exit 1; }
    echo "=== [$scene] stage3 timing (GPU$gpu) ==="
    CUDA_VISIBLE_DEVICES=$gpu $PY n1_stage3_timing.py $scene || { echo "TIMING FAIL $scene"; exit 1; }
    echo "=== [$scene] PIPELINE COMPLETE ==="
  } > logs/stage23_$scene.log 2>&1
}

run_scene room 0 &
P0=$!
run_scene bicycle 1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "room rc=$R0 bicycle rc=$R1"
echo "=== room tail ==="; tail -6 logs/stage23_room.log
echo "=== bicycle tail ==="; tail -6 logs/stage23_bicycle.log
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader > cont_stage23_end.csv
exit $(( R0 + R1 ))
