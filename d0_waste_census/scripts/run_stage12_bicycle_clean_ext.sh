#!/bin/bash
# D0 Stage 2 extension: +3 frozen-recipe clean bicycle 30K replicates.
# Pre-registered in artifacts/D0_BICYCLE_FINALN_SCREEN_ESCALATION_PREREG.json:
# the bicycle final_N 2x screen fired on the 3-run envelope; the envelope
# estimate is refined with more clean draws and the SAME screen re-applied.
# Launch command is byte-identical to the run_stage12.sh clean arm.
set -u
ROOT=/root/epic05_d0
PYW=/root/epic05_py_wrapper.sh
G=2
FROZEN_TRAINER=/root/epic05_p12/publication_trainer_p12.py
mkdir -p "$ROOT/logs" "$ROOT/runs"

bash "$ROOT/scripts/gpu_gate.sh" "$G" 10 30 || { echo "GATE FAIL - aborting clean-ext"; exit 3; }

run_one () {
  local id="$1"; shift
  local outdir="$ROOT/runs/$id"
  local log="$ROOT/logs/$id.log"
  mkdir -p "$outdir"
  echo "== $id start $(date -Is)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $G > "$outdir/gpu_apps_before.txt" || true
  CUDA_VISIBLE_DEVICES=$G timeout 14400 "$PYW" "$FROZEN_TRAINER" --arm c0 --scene bicycle --iterations 30000 \
      --outdir "$outdir" --gpu 0 --lpips --final-eval all \
      --timing-grade FUNCTIONAL_ONLY --eps2d 0.3 --loss fused > "$log" 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $G > "$outdir/gpu_apps_after.txt" || true
  echo "$rc" > "$outdir/trainer_rc.txt"
  date -Is > "$outdir/finished_at.txt"
  [ $rc -eq 0 ] && touch "$outdir/STAGE12_DONE"
  echo "== $id done rc=$rc $(date -Is)"
  return 0
}

run_one CLEAN3_bicycle
run_one CLEAN4_bicycle
run_one CLEAN5_bicycle

echo "== stage12 bicycle clean-ext complete $(date -Is)"
