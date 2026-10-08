#!/bin/bash
# watch_and_run.sh - guard-safe idle-GPU watcher for the production CoMax renderer.
# Polls allowed GPUs [1-5]; FROZEN idle gate util<=2% AND mem<=512MiB for 20s stable.
# Protected GPUs [0,6,7] never selected. Co-resident train_tf.py NEVER touched.
# NEVER kills/renices/resets/sudo. If no idle card, keep polling.
#
# E2-prep fix (2026-10-06): the previous version launched the dense harness
# directly, so require_guarded_process() raised immediately (no COMAX0_GUARD_PROOF)
# => silent CORRECTNESS FAIL. It now launches scripts/run_e2_correctness.py through
# scripts/run_on_mx.py, which writes the guard proof and applies the CPU->CUDA
# device fixups. Correctness-only; NO timing. If the E2-prep anchor already exists,
# the watcher stands by (E2 timing is NOT legitimately runnable until the reducer's
# strict evidence set exists), so it neither burns budget nor silently fails.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH"
PY="/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python"
ALLOWED=(1 2 3 4 5)
IDLE_UTIL=2
IDLE_MEM=512
IDLE_SECS=20
POLL_SECS=30
RUN_DIR="$ROOT/run"
mkdir -p "$RUN_DIR"
LOG="$ROOT/logs/watch.log"
PREP_ANCHOR="/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_E2_CORRECTNESS_PREP/run/E2_CORRECTNESS_SUMMARY.json"
echo "[watch] start $(date) (e2-prep fixed watcher)" | tee -a "$LOG"

selected=""; idle_since=0; best_uuid=""
while true; do
  ts=$(date +%s)
  best=""; best_util=9999; best_mem=999999
  while IFS="," read -r idx uuid mempart utilpart; do
    idx=$(echo "$idx" | tr -d " ")
    uuid=$(echo "$uuid" | tr -d " ")
    mem=$(echo "$mempart" | tr -d " ")
    util=$(echo "$utilpart" | tr -d " ")
    [ -z "$idx" ] && continue
    allow=0; for a in "${ALLOWED[@]}"; do if [ "$a" = "$idx" ]; then allow=1; fi; done
    [ "$allow" = "1" ] || continue
    if [ "$util" -le "$IDLE_UTIL" ] && [ "$mem" -le "$IDLE_MEM" ]; then
      if [ "$util" -lt "$best_util" ] || { [ "$util" -eq "$best_util" ] && [ "$mem" -lt "$best_mem" ]; }; then
        best="$idx"; best_util=$util; best_mem=$mem; best_uuid=$uuid
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  if [ -n "$best" ]; then
    if [ -z "$selected" ]; then selected=$best; idle_since=$ts; fi
    stable=$((ts - idle_since))
    echo "[watch] cand GPU$best util=${best_util}% mem=${best_mem}MB stable=${stable}s" | tee -a "$LOG"
    if [ "$stable" -ge "$IDLE_SECS" ]; then
      if [ -f "$PREP_ANCHOR" ] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('all_pass') else 1)" "$PREP_ANCHOR"; then
        echo "[watch] idle GPU$best but E2-prep anchor present and E2 timing not legitimately runnable; standby" | tee -a "$LOG"
        selected=""; idle_since=0; sleep "$POLL_SECS"; continue
      fi
      echo "[watch] LAUNCH wrapper via run_on_mx on GPU$best ($best_uuid) stable ${stable}s" | tee -a "$LOG"
      RUNID="e2_prep_auto_$(date +%Y%m%d_%H%M%S)"
      "$PY" "$PKG/scripts/run_on_mx.py" --run-id "$RUNID" --gpu-uuid "$best_uuid" --stage e2 --entry scripts/run_e2_correctness.py --max-minutes 10 >> "$LOG" 2>&1
      rc=$?
      echo "[watch] wrapper rc=$rc -> $ROOT/run/$RUNID/e2/E2_CORRECTNESS_SUMMARY.json" | tee -a "$LOG"
      if [ $rc -eq 0 ]; then
        echo "[watch] E2-prep correctness PASS. E2 timing still requires reducer evidence; standby." | tee -a "$LOG"
      else
        echo "[watch] E2-prep correctness FAIL. STOP. Do not profile an incorrect fast path." | tee -a "$LOG"
      fi
      exit $rc
    fi
  else
    echo "[watch] no idle allowed GPU; reset" | tee -a "$LOG"
    selected=""; idle_since=0
  fi
  sleep "$POLL_SECS"
done
