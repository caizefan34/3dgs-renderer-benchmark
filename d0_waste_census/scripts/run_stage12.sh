#!/bin/bash
# D0 Stage 1/2: clean replay + traced 30K runs per scene, physical GPU 2.
# Usage: run_stage12.sh <room|bicycle>
set -u
ROOT=/root/epic05_d0
PYW=/root/epic05_py_wrapper.sh
G=2
FROZEN_TRAINER=/root/epic05_p12/publication_trainer_p12.py
D0_TRAINER=$ROOT/scripts/d0_trainer.py
WD=/root/3dgs-publication-epic05/watchdog
SCENE="${1:?usage: run_stage12.sh <room|bicycle>}"
mkdir -p "$ROOT/logs" "$ROOT/runs" "$ROOT/artifacts"

snapshot () {   # snapshot <tag>
  local tag="$1"
  local f="$ROOT/artifacts/contamination_${tag}.txt"
  {
    echo "tag=$tag date=$(date -Is)"
    echo "gpu_uuid_train=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G)"
    echo "gpu2_mem_util=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $G)"
    echo "-- compute apps (all gpus) --"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    echo "-- our python procs --"
    ps -eo pid,etimes,comm,args | grep '[p]ublication_trainer\|[d]0_trainer' || true
    echo "-- wrapper pid file gpu$G --"
    cat "$WD/current_pids/gpu$G.txt" 2>/dev/null || true
    echo "-- watchdog compute_apps tail --"
    tail -3 "$WD/compute_apps.jsonl" 2>/dev/null || true
    echo "-- watchdog thermal tail --"
    tail -2 "$WD/thermal.jsonl" 2>/dev/null || true
  } > "$f"
  echo "  contamination snapshot -> $f"
}

run_one () {   # run_one <id> <trainer> <extra...>
  local id="$1"; shift
  local trainer="$1"; shift
  local outdir="$ROOT/runs/$id"
  local log="$ROOT/logs/$id.log"
  mkdir -p "$outdir"
  echo "== $id start $(date -Is)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $G > "$outdir/gpu_apps_before.txt" || true
  CUDA_VISIBLE_DEVICES=$G timeout 14400 "$PYW" "$trainer" "$@" \
      --outdir "$outdir" --gpu 0 --lpips --final-eval all \
      --timing-grade FUNCTIONAL_ONLY --eps2d 0.3 --loss fused > "$log" 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $G > "$outdir/gpu_apps_after.txt" || true
  echo "$rc" > "$outdir/trainer_rc.txt"
  date -Is > "$outdir/finished_at.txt"
  [ $rc -eq 0 ] && touch "$outdir/STAGE12_DONE"
  echo "== $id done rc=$rc $(date -Is)"
  return $rc
}

snapshot "stage12_${SCENE}_start"
run_one "CLEAN_${SCENE}" "$FROZEN_TRAINER" --arm c0 --scene "$SCENE" --iterations 30000
rc_clean=$?
# Second clean replay: the frozen renderer is run-to-run nondeterministic, so
# the trace-validity spread rule needs an intrinsic 30K envelope with three
# clean samples (P12A reference + CLEAN + CLEAN2) against the traced run.
run_one "CLEAN2_${SCENE}" "$FROZEN_TRAINER" --arm c0 --scene "$SCENE" --iterations 30000
rc_clean2=$?
run_one "TRACE_${SCENE}" "$D0_TRAINER"     --arm c0 --scene "$SCENE" --iterations 30000 --d0-trace
rc_trace=$?
snapshot "stage12_${SCENE}_end"

echo "== stage12 $SCENE complete clean_rc=$rc_clean clean2_rc=$rc_clean2 trace_rc=$rc_trace $(date -Is)"
ls -la "$ROOT/runs/CLEAN_${SCENE}" "$ROOT/runs/CLEAN2_${SCENE}" "$ROOT/runs/TRACE_${SCENE}" 2>/dev/null | head -40
exit $(( rc_clean + rc_clean2 + rc_trace ))
