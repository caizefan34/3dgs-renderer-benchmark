#!/bin/bash
# P-TRANSITION batch: 12 preregistered units (bicycle/garden x 5K/14K x 3 new seeds)
# Single clean GPU cohort (GPU 3). Appends contamination records per unit.
set -u
ROOT=/root/epic05_ptransition
PY=/root/miniforge3/envs/gsplat/bin/python
GPU=${PT_GPU:-3}
LOGDIR=$ROOT/logs
RESULTS=$ROOT/results
SNAPDIR=/root/epic05_grc3d/inputs/snapshots
P12RUNS=/root/epic05_p12/runs/p12a
mkdir -p "$LOGDIR" "$RESULTS"
CONTAM=$LOGDIR/P_TRANSITION_CONTAMINATION.log

echo "=== P-TRANSITION BATCH START $(date -u +%Y-%m-%dT%H:%M:%SZ) GPU=$GPU" >> "$CONTAM"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader >> "$CONTAM"

run_unit () {
  scene=$1; iter=$2; seed=$3
  snap=$SNAPDIR/snap_${scene}_$(printf %06d $iter).pt
  camseq=$P12RUNS/p12a_gmainf_s42_${scene}/camera_sequence.npy
  out=$RESULTS/transition_${scene}_${iter}_s${seed}.json
  log=$LOGDIR/unit_${scene}_${iter}_s${seed}.log
  echo "--- UNIT $scene iter=$iter seed=$seed start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CONTAM"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >> "$CONTAM"
  ( cd "$ROOT" && "$PY" run_transition.py \
      --gx-path /root/epic05_grc3d/scripts \
      --scene "$scene" --snapshot "$snap" --camera-seq "$camseq" \
      --out "$out" --gpu "$GPU" --iteration "$iter" --seed "$seed" ) > "$log" 2>&1
  rc=$?
  echo "--- UNIT $scene iter=$iter seed=$seed rc=$rc end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CONTAM"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >> "$CONTAM"
  return $rc
}

fail=0
for scene in bicycle garden; do
  for iter in 5000 14000; do
    for seed in 271828 314159 161803; do
      if ! run_unit "$scene" "$iter" "$seed"; then
        fail=$((fail+1))
        echo "!!! UNIT FAILED $scene $iter $seed" >> "$CONTAM"
      fi
    done
  done
done
echo "=== P-TRANSITION BATCH END $(date -u +%Y-%m-%dT%H:%M:%SZ) failures=$fail" >> "$CONTAM"

( cd "$ROOT" && "$PY" analyze_transition.py --inputs results --out P_TRANSITION_AGG.json ) >> "$LOGDIR/analyze.log" 2>&1
echo "ANALYZE rc=$?" >> "$CONTAM"
