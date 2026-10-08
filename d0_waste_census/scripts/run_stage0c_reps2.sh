#!/bin/bash
# D0 Stage 0c replicates (traced side): repeat runs of the INSTRUMENTED trainer
# to measure the traced worktree's own run-to-run spread.  Together with the
# clean replicates this gives the symmetric comparison
#   d(clean,clean) vs d(traced,traced) vs d(clean,traced)
# for the spread rule.  Same traced launch command as run_stage0c_trace.sh.
set -u
ROOT=/root/epic05_d0
PYW=/root/epic05_py_wrapper.sh
G=2
D0_TRAINER=$ROOT/scripts/d0_trainer.py
mkdir -p "$ROOT/logs" "$ROOT/runs"

run_one () {
  local id="$1"; shift
  local outdir="$ROOT/runs/$id"
  local log="$ROOT/logs/$id.log"
  rm -rf "$outdir"; mkdir -p "$outdir"
  echo "== $id start $(date -Is)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $G > "$outdir/gpu_apps_before.txt" || true
  CUDA_VISIBLE_DEVICES=$G timeout 3600 "$PYW" "$D0_TRAINER" "$@" \
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

run_one T3room     --arm c0 --scene room    --iterations 3000 --d0-trace
run_one T3bicycle  --arm c0 --scene bicycle --iterations 3000 --d0-trace
run_one T3room2    --arm c0 --scene room    --iterations 3000 --d0-trace
run_one T3room3    --arm c0 --scene room    --iterations 3000 --d0-trace
run_one T3bicycle2 --arm c0 --scene bicycle --iterations 3000 --d0-trace

echo "== stage0c traced replicates complete $(date -Is)"
