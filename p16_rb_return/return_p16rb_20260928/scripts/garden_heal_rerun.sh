#!/bin/bash
# P16-RB garden heal rerun (operator-approved 2026-09-28). Runs only after
# heal_garden_data.sh logged HEAL_DONE.
#   A) archive the flat-data garden evidence (renames only, nothing deleted)
#   B) re-run garden on the healed dataset for all four external methods, sequential on
#      physical GPU 2, replicating run_stage2_triad.sh exactly (wrapper + vram sampler +
#      before/after captures + finalize_matrix.sh stage2 <M>:garden)
#   C) re-run the frozen P12 C0 garden (exact archived recipe) + common-split render +
#      canonical metrics + P16_RUN.json rebuild via p16_build_p12_record.py
set -u
D=/root/epic05_p16rb
GPU=2
WRAP=$D/bundle/p16rb_wrap_run.py
LOG=$D/logs/garden_heal_rerun.log
: > $LOG
log() { echo "[$(date -Is)] $*" | tee -a $LOG; }
. $D/scripts/p16_recipes.sh

grep -q HEAL_DONE $D/logs/garden_heal.log || { log "REFUSING: heal not done"; exit 1; }

EV=$D/backups/garden_flatdata_evidence
mkdir -p "$EV/logs"

capture() {
  local L=$1
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
    --format=csv,noheader > ${L}.gpu_before.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_before.csv
}
capture_after() {
  local L=$1
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
    --format=csv,noheader > ${L}.gpu_after.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_after.csv
}

# ---------- A: archive flat-data garden evidence ----------
for M in FasterGS FasterGSFused FastGS SADGS P12_C0; do
  R=$D/runs/$M/garden
  if [ -d "$R" ] && [ ! -d "${R}_flatdata_corrupt" ]; then
    mv "$R" "${R}_flatdata_corrupt" && log "archived runs/$M/garden -> runs/$M/garden_flatdata_corrupt"
  fi
done
for G in $D/logs/stage2_FasterGS_garden* $D/logs/stage2_FasterGSFused_garden* \
         $D/logs/stage2_FastGS_garden* $D/logs/stage2_SADGS_garden* \
         $D/logs/P12_C0_garden.* $D/logs/P12_C0_common_garden*; do
  [ -e "$G" ] && { mv "$G" "$EV/logs/"; log "archived log $(basename "$G")"; }
done
for M in FastGS SADGS; do
  J=$D/src/$M/output/p16_${M}_garden
  if [ -d "$J" ] && [ ! -d "${J}_flatdata_corrupt" ]; then
    mv "$J" "${J}_flatdata_corrupt" && log "archived src/$M/output/p16_${M}_garden"
  fi
done

# ---------- B: external-method reruns (healed data, GPU 2) ----------
SC=garden
for M in FasterGS FasterGSFused FastGS SADGS; do
  ENV=$(native_env $M); N=$(native_n $M)
  PY=/root/miniforge3/envs/$ENV/bin/python
  OUT=$D/runs/$M/$SC
  mkdir -p $OUT
  TAG=${M}_${SC}
  L=$D/logs/stage2_${TAG}
  capture $L
  $PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/stage2_${TAG}.vram.csv \
    --duration 28800 > $D/logs/stage2_${TAG}.sampler.log 2>&1 &
  SAMP=$!
  log "$M $SC train start (healed)"
  if [ "$M" = "FasterGS" ] || [ "$M" = "FasterGSFused" ]; then
    LOWER=$(echo $M | tr 'A-Z' 'a-z')
    CFG=$D/configs/p16_${LOWER}_${SC}_1920.yaml
    $PY $WRAP --method $M --scene $SC --cwd $D/src/nerficg \
      --command "PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python scripts/train.py -c $CFG" \
      --out $OUT/wrapper_train.json --log $D/logs/stage2_${TAG}_train.stdout.log
    log "$M $SC train rc=$?"
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
    capture_after $L
  else
    JOB=p16_${M}_${SC}
    $PY $WRAP --method $M --scene $SC --cwd $D/src/$M \
      --command "CUDA_VISIBLE_DEVICES=$GPU OAR_JOB_ID=$JOB PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python $(graphdeco_recipe $M $SC $N)" \
      --out $OUT/wrapper_train.json --log $D/logs/stage2_${TAG}_train.stdout.log
    log "$M $SC train rc=$?"
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
    capture_after $L
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
  echo "STAGE2_${TAG}_HEALED_DONE $(date -Is)" >> $D/logs/stage2.log
  bash $D/scripts/finalize_matrix.sh stage2 ${M}:${SC}
  cp $D/logs/finalize_stage2.log $EV/finalize_stage2_${M}_garden_healed.log
  if [ -f $D/runs/$M/$SC/P16_RUN.json ]; then
    $PY - "$D/runs/$M/$SC/P16_RUN.json" <<'PYEOF' >> $LOG 2>&1
import json, sys
r = json.load(open(sys.argv[1]))
print("HEALED_RECORD", r["method"], r["scene"], r["metrics_common"], "clean=", r["clean"])
PYEOF
    log "FINALIZED ${M} garden (healed)"
  else
    log "FINALIZE_FAIL ${M} garden (healed)"
  fi
done

# ---------- C: P12 C0 garden rerun (exact archived recipe) ----------
OUT=$D/runs/P12_C0/garden
mkdir -p "$OUT"
L=$D/logs/P12_C0_garden_healed
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
  --format=csv,noheader -i $GPU > ${L}.gpu_before.csv
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader > ${L}.procs_before.csv
uptime > ${L}.uptime_before.txt
T0=$(date -Is)
CUDA_VISIBLE_DEVICES=$GPU /root/epic05_py_wrapper.sh /root/epic05_p12/publication_trainer_p12.py \
  --arm c0 --scene garden --iterations 30000 --outdir "$OUT" --gpu $GPU --lpips \
  --final-eval all --checkpoints --timing-grade PUBLICATION --eps2d 0.3 --loss fused \
  > ${L}.stdout.log 2>&1
RC=$?
T1=$(date -Is)
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
  --format=csv,noheader -i $GPU > ${L}.gpu_after.csv
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader > ${L}.procs_after.csv
uptime > ${L}.uptime_after.txt
echo "garden rc=$RC T0=$T0 T1=$T1" > ${L}.triad.log
log "$(cat ${L}.triad.log)"

# P12 common-split render (garden only), replicating run_p12_common_eval.sh
OUTE=$D/runs/P12_C0/garden/common_eval
CKPT=$D/runs/P12_C0/garden/ckpt_30000.pt
EXP=$D/artifacts/expected_names_garden.json
if [ ! -f "$CKPT" ]; then log "P12 CKPT MISSING - aborting P12 part"; exit 7; fi
mkdir -p "$OUTE"
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
  --format=csv,noheader -i $GPU > ${L}.gpu_before.csv
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader > ${L}.procs_before.csv
T2=$(date -Is)
/root/miniforge3/envs/p16_fastergs/bin/python $D/scripts/p16_expected_names.py garden "$EXP" \
  >> ${L}.eval.stdout.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU /root/miniforge3/envs/gsplat/bin/python $D/scripts/p16_p12_render_common.py \
  --scene garden --ckpt "$CKPT" --outdir "$OUTE" --gpu $GPU \
  --expected-names-file "$EXP" >> ${L}.eval.stdout.log 2>&1
RC2=$?
T3=$(date -Is)
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
  --format=csv,noheader -i $GPU > ${L}.gpu_after.csv
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader > ${L}.procs_after.csv
echo "garden common-eval rc=$RC2 T0=$T2 T1=$T3" >> ${L}.triad.log
log "garden common-eval rc=$RC2"

# canonical common metrics (GPU 3, metric card; same pipeline as every method)
( cd $D && CUDA_VISIBLE_DEVICES=3 /root/miniforge3/envs/p16_fastergs/bin/python \
    $D/scripts/p16_common_metrics.py --method P12_C0 --scene garden \
    --rgb-dir runs/P12_C0/garden/common_eval/images \
    --out runs/P12_C0/garden/metrics_common.json \
    --gt-dir $D/common_data/mipnerf360/garden/images \
    --expected-names $D/artifacts/expected_names_garden.json ) >> ${L}.eval.stdout.log 2>&1
log "canonical metrics rc=$?"

/root/miniforge3/envs/p16_fastergs/bin/python $D/scripts/p16_build_p12_record.py \
  --scene garden \
  --rundir $D/runs/P12_C0/garden \
  --triad-log ${L}.triad.log \
  --metrics-json $D/runs/P12_C0/garden/metrics_common.json \
  --out $D/runs/P12_C0/garden/P16_RUN.json \
  --source-commit P12_C0_FROZEN@2c6d9685b0eca852f6749ff4d64dc5a3ef942237 \
  --log-prefix $D/logs/P12_C0_garden_healed \
  --trainer /root/epic05_p12/publication_trainer_p12.py >> $LOG 2>&1
log "P12 record rebuild rc=$?"

echo "GARDEN_HEAL_RERUN_DONE $(date -Is)" | tee -a $LOG
