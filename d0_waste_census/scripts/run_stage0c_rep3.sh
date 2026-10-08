#!/bin/bash
# D0 Stage 0c rep-3: pre-specified replication to resolve the marginal
# max-envelope exceedance of the run-1 smoke ensemble
# (see artifacts/D0_TRACE_VALIDITY_SCREEN_PREREG.json).  Pre-specified set:
# +3 traced room, +2 traced bicycle, +2 clean bicycle.  Launch commands are
# identical to run_stage0c_trace.sh / run_stage0c_reps.sh; no rule is changed.
set -u
ROOT=/root/epic05_d0
PYW=/root/epic05_py_wrapper.sh
G=2
FROZEN_TRAINER=/root/epic05_p12/publication_trainer_p12.py
D0_TRAINER=$ROOT/scripts/d0_trainer.py
mkdir -p "$ROOT/logs" "$ROOT/runs"

bash "$ROOT/scripts/gpu_gate.sh" "$G" 10 30 || { echo "GATE FAIL - aborting rep3"; exit 3; }

run_one () {
  local id="$1"; shift
  local trainer="$1"; shift
  local outdir="$ROOT/runs/$id"
  local log="$ROOT/logs/$id.log"
  rm -rf "$outdir"; mkdir -p "$outdir"
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

run_one T3room4    "$D0_TRAINER"     --arm c0 --scene room    --iterations 3000 --d0-trace
run_one T3room5    "$D0_TRAINER"     --arm c0 --scene room    --iterations 3000 --d0-trace
run_one T3room6    "$D0_TRAINER"     --arm c0 --scene room    --iterations 3000 --d0-trace
run_one T3bicycle3 "$D0_TRAINER"     --arm c0 --scene bicycle --iterations 3000 --d0-trace
run_one T3bicycle4 "$D0_TRAINER"     --arm c0 --scene bicycle --iterations 3000 --d0-trace
run_one C3bicycle4 "$FROZEN_TRAINER" --arm c0 --scene bicycle --iterations 3000
run_one C3bicycle5 "$FROZEN_TRAINER" --arm c0 --scene bicycle --iterations 3000

echo "== stage0c rep3 complete $(date -Is)"
