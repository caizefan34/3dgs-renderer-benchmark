#!/bin/bash
# P16-RB: after CURVES_ALL_DONE -> uniform v2 curve rebuild -> verify gate -> correction
# artifact -> aggregate table -> return packaging. Stops before the report step (manual).
set -u
D=/root/epic05_p16rb
PY=/root/miniforge3/envs/gsplat/bin/python
LOG=$D/logs/finalize_15.log
log() { echo "[$(date -Is)] $*" | tee -a $LOG; }
: > $LOG

log "waiting for FINAL_RERUN3_DONE"
for i in $(seq 1 240); do
  grep -q FINAL_RERUN3_DONE $D/logs/final_rerun3.log 2>/dev/null && break
  sleep 60
done
grep -q FINAL_RERUN3_DONE $D/logs/final_rerun3.log || { log "TIMEOUT waiting final_rerun3"; exit 1; }
log "curves queue done"

log "step 1: rebuild curves v2 (archive v1 + rebuild 12 + verify gate)"
bash $D/scripts/p16_rebuild_curves.sh > $D/logs/curve_rebuild_v2.log 2>&1
RC=$?
grep -E "^ARCHIVED_V1|^REBUILD_|^OK |^BAD |VERIFY_CURVES_BAD|CURVES_REBUILD_DONE" $D/logs/curve_rebuild_v2.log | tail -30 | tee -a $LOG
if [ $RC -ne 0 ]; then log "CURVES_VERIFY_GATE_FAIL rc=$RC"; exit 1; fi
log "REBUILD_AND_VERIFY_OK"

log "step 2: correction artifact (v1 vs v2 evidence)"
$PY $D/scripts/p16_curve_correction_record.py 2>&1 | tee -a $LOG

log "step 3: aggregate table"
bash $D/scripts/p16_aggregate_table.sh > $D/logs/aggregate_15.log 2>&1
log "aggregate rc=$?"
tail -40 $D/logs/aggregate_15.log | tee -a $LOG

log "step 4: return packaging (pre-report)"
$PY $D/scripts/p16_package_return.py 2>&1 | tee -a $LOG

log "P16_FINALIZE_15_DONE"
