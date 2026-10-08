#!/bin/bash
# P16-RB Stage 2: bicycle + garden for the methods that passed the room pilot.
# SEQUENTIAL on physical A100 GPU 2, same discipline as Stage 1 (wrapper + sampler +
# before/after process captures). Per-scene author recipes from p16_recipes.sh.
#
# usage: run_stage2_triad.sh FasterGS FasterGSFused FastGS SADGS
set -u
D=/root/epic05_p16rb
GPU=2
WRAP=$D/bundle/p16rb_wrap_run.py
S2=$D/logs/stage2.log
: > $S2
log() { echo "[$(date -Is)] $*" | tee -a $S2; }
. $D/scripts/p16_recipes.sh

capture() {
  local L=$D/logs/stage2_$1
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
    --format=csv,noheader > ${L}.gpu_before.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_before.csv
}
capture_after() {
  local L=$D/logs/stage2_$1
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
    --format=csv,noheader > ${L}.gpu_after.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_after.csv
}

for M in "$@"; do
  ENV=$(native_env $M); N=$(native_n $M)
  PY=/root/miniforge3/envs/$ENV/bin/python
  for SC in bicycle garden; do
    OUT=$D/runs/$M/$SC
    mkdir -p $OUT
    TAG=${M}_${SC}
    capture $TAG
    $PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/stage2_${TAG}.vram.csv \
      --duration 28800 > $D/logs/stage2_${TAG}.sampler.log 2>&1 &
    SAMP=$!
    log "$M $SC train start"
    if [ "$M" = "FasterGS" ] || [ "$M" = "FasterGSFused" ]; then
      LOWER=$(echo $M | tr 'A-Z' 'a-z')
      CFG=$D/configs/p16_${LOWER}_${SC}_1920.yaml
      $PY $WRAP --method $M --scene $SC --cwd $D/src/nerficg \
        --command "PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python scripts/train.py -c $CFG" \
        --out $OUT/wrapper_train.json --log $D/logs/stage2_${TAG}_train.stdout.log
      log "$M $SC train rc=$?"
      kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
      capture_after $TAG
    else
      JOB=p16_${M}_${SC}
      $PY $WRAP --method $M --scene $SC --cwd $D/src/$M \
        --command "CUDA_VISIBLE_DEVICES=$GPU OAR_JOB_ID=$JOB PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python $(graphdeco_recipe $M $SC $N)" \
        --out $OUT/wrapper_train.json --log $D/logs/stage2_${TAG}_train.stdout.log
      log "$M $SC train rc=$?"
      kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
      capture_after $TAG
      log "$M $SC render start (untimed)"
      if [ "$M" = "SADGS" ]; then
        RCMD="python render.py -m output/$JOB --skip_train --iteration $N"
      else
        RCMD="python render.py -m output/$JOB --skip_train"
      fi
      $PY $WRAP --method $M --scene $SC --cwd $D/src/$M \
        --command "CUDA_VISIBLE_DEVICES=$GPU PATH=/root/miniforge3/envs/$ENV/bin:\$PATH $RCMD" \
        --out $OUT/wrapper_render.json --log $D/logs/stage2_${TAG}_render.stdout.log
      log "$M $SC render rc=$?"
    fi
    echo "STAGE2_${TAG}_DONE $(date -Is)" | tee -a $S2
  done
done
echo "STAGE2_ALL_DONE $(date -Is)" | tee -a $S2
