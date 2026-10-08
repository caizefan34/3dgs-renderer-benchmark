#!/bin/bash
# D0 Stage 0c smoke: clean vs traced 3000-iter runs, room (+clean repeat) and bicycle.
# All runs on physical GPU 2 (training GPU); launched via the frozen EPIC wrapper.
set -u
ROOT=/root/epic05_d0
PYW=/root/epic05_py_wrapper.sh
G=2
FROZEN_TRAINER=/root/epic05_p12/publication_trainer_p12.py
D0_TRAINER=$ROOT/scripts/d0_trainer.py
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

run_one C3room1   "$FROZEN_TRAINER" --arm c0 --scene room    --iterations 3000
run_one C3room2   "$FROZEN_TRAINER" --arm c0 --scene room    --iterations 3000
run_one T3room    "$D0_TRAINER"     --arm c0 --scene room    --iterations 3000 --d0-trace
run_one C3bicycle "$FROZEN_TRAINER" --arm c0 --scene bicycle --iterations 3000
run_one T3bicycle "$D0_TRAINER"     --arm c0 --scene bicycle --iterations 3000 --d0-trace

echo "== stage0c batch complete $(date -Is)"
ls -la "$ROOT/runs"
