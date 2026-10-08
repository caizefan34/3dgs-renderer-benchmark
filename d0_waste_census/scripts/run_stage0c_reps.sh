#!/bin/bash
# D0 Stage 0c replicates: extra CLEAN frozen-recipe 3000-iter smoke runs.
# Purpose: the frozen C0 renderer is run-to-run nondeterministic (verified:
# iteration-1 loss bit-identical, then ~1e-7 backward-atomic noise that
# amplifies at densification thresholds).  The trace-validity spread rule
# needs an intrinsic-spread envelope with more than two samples, so these
# replicates characterise the clean ensemble around the traced runs.
# Recipe identical to run_stage0c.sh clean arms; nothing is changed.
set -u
ROOT=/root/epic05_d0
PYW=/root/epic05_py_wrapper.sh
G=2
FROZEN_TRAINER=/root/epic05_p12/publication_trainer_p12.py
mkdir -p "$ROOT/logs" "$ROOT/runs"

run_one () {
  local id="$1"; shift
  local trainer="$1"; shift
  local outdir="$ROOT/runs/$id"
  local log="$ROOT/logs/$id.log"
  mkdir -p "$outdir"
  echo "== $id start $(date -Is)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $G > "$outdir/gpu_apps_before.txt" || true
  CUDA_VISIBLE_DEVICES=$G timeout 3600 "$PYW" "$trainer" "$@" \
      --outdir "$outdir" --gpu 0 --lpips --final-eval all \
      --timing-grade FUNCTIONAL_ONLY --eps2d 0.3 --loss fused > "$log" 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $G > "$outdir/gpu_apps_after.txt" || true
  echo "$rc" > "$outdir/trainer_rc.txt"
  date -Is > "$outdir/finished_at.txt"
  if [ $rc -eq 0 ]; then touch "$outdir/STAGE0C_DONE"; fi
  echo "== $id done rc=$rc $(date -Is)"
  return 0
}

run_one C3room3    "$FROZEN_TRAINER" --arm c0 --scene room    --iterations 3000
run_one C3room4    "$FROZEN_TRAINER" --arm c0 --scene room    --iterations 3000
run_one C3room5    "$FROZEN_TRAINER" --arm c0 --scene room    --iterations 3000
run_one C3room6    "$FROZEN_TRAINER" --arm c0 --scene room    --iterations 3000
run_one C3bicycle2 "$FROZEN_TRAINER" --arm c0 --scene bicycle --iterations 3000
run_one C3bicycle3 "$FROZEN_TRAINER" --arm c0 --scene bicycle --iterations 3000

echo "== stage0c replicates complete $(date -Is)"
