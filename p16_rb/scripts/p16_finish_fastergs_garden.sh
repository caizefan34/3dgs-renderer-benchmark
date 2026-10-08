#!/bin/bash
# P16-RB: rerun the failed FasterGS garden stage-2 cell after the triad chain drains,
# then finalize that pair. Sequence: CHAIN_STAGE2_DONE -> this script -> GATE_CLEARED.
# Replicates run_stage2_triad.sh exactly (wrapper + sampler + before/after captures);
# the training card is physical GPU 2, metric evaluation runs on GPU 3.
set -u
D=/root/epic05_p16rb
GPU=2
WRAP=$D/bundle/p16rb_wrap_run.py
LOG=$D/logs/garden_retry.log
. $D/scripts/p16_recipes.sh
log() { echo "[$(date -Is)] $*" | tee -a $LOG; }
: > $LOG

log "waiting for CHAIN_STAGE2_DONE"
for i in $(seq 1 480); do
  grep -q "CHAIN_STAGE2_DONE" $D/logs/chain_stage2.log 2>/dev/null && break
  sleep 60
done
grep -q "CHAIN_STAGE2_DONE" $D/logs/chain_stage2.log || { log "TIMEOUT waiting chain_stage2"; exit 1; }

log "chain done; waiting for GPU $GPU to drain"
UUID=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F", " -v g=$GPU '$1==g {print $2}')
N=1
for i in $(seq 1 120); do
  N=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID" || true)
  [ "$N" -eq 0 ] && break
  sleep 30
done
[ "$N" -eq 0 ] || { log "TIMEOUT waiting GPU drain (still $N apps)"; exit 1; }

M=FasterGS; SC=garden; TAG=${M}_${SC}
ENV=$(native_env $M); PY=/root/miniforge3/envs/$ENV/bin/python
OUT=$D/runs/$M/$SC
mkdir -p $OUT

[ -f $D/logs/stage2_${TAG}_train.stdout.log ] && cp -a $D/logs/stage2_${TAG}_train.stdout.log $D/logs/stage2_${TAG}_train.stdout.failed_attempt.log

env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader > $D/logs/stage2_${TAG}.gpu_before.csv
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $D/logs/stage2_${TAG}.procs_before.csv
$PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/stage2_${TAG}.vram.csv --duration 28800 > $D/logs/stage2_${TAG}.sampler.log 2>&1 &
SAMP=$!
CFG=$D/configs/p16_fastergs_garden_1920.yaml
log "$M $SC retry train start"
$PY $WRAP --method $M --scene $SC --cwd $D/src/nerficg \
  --command "PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python scripts/train.py -c $CFG" \
  --out $OUT/wrapper_train.json --log $D/logs/stage2_${TAG}_train.stdout.log
RC=$?
log "$M $SC retry train rc=$RC"
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader > $D/logs/stage2_${TAG}.gpu_after.csv
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $D/logs/stage2_${TAG}.procs_after.csv

if [ $RC -ne 0 ]; then
  log "RETRY_TRAIN_FAIL rc=$RC"
  echo "GARDEN_RETRY_FAILED $(date -Is)" >> $LOG
  exit 1
fi

log "finalize $M:$SC"
bash $D/scripts/finalize_matrix.sh stage2 ${M}:${SC} 2>&1 | tee -a $LOG
if [ -f $OUT/P16_RUN.json ]; then
  log "P16_RUN.json present"
else
  log "P16_RUN.json MISSING after finalize"
  echo "GARDEN_RETRY_FAILED $(date -Is)" >> $LOG
  exit 1
fi
echo "GARDEN_RETRY_DONE $(date -Is)" >> $LOG
log "done"
