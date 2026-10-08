#!/bin/bash
# D0 Stage 0c repair: rerun traced replicates that failed on the audit
# tiles/opacity bug (fixed in d0_trace.py; training path was unaffected).
# Own clean-GPU gate; same traced launch command as run_stage0c_trace.sh.
# Usage: run_stage0c_repair.sh [id ...]   (default: T3room T3bicycle T3room2)
set -u
ROOT=/root/epic05_d0
PYW=/root/epic05_py_wrapper.sh
G=2
D0_TRAINER=$ROOT/scripts/d0_trainer.py
IDS="${*:-T3room T3bicycle T3room2}"
mkdir -p "$ROOT/logs" "$ROOT/runs"

bash "$ROOT/scripts/gpu_gate.sh" "$G" 10 30

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

for id in $IDS; do
  scene=$(echo "$id" | sed 's/^T3//; s/[0-9]*$//')
  run_one "$id" --arm c0 --scene "$scene" --iterations 3000 --d0-trace
done

echo "== stage0c repair complete $(date -Is)"
