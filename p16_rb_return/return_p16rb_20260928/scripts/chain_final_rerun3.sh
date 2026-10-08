#!/bin/bash
# P16-RB RERUN 3: curves phase only, after the run_curves.sh CRLF deployment fix (2026-09-28).
# rerun-2 (chain_final_rerun2.sh) is in its profiles phase on GPU 2; this script waits for the
# FINAL_RERUN2_DONE marker (or 3 consecutive checks with no run_profiles process) before it
# touches GPU 2, then runs the corrected 12-pair TTQ curve queue.
set -u
D=/root/epic05_p16rb
LOG=$D/logs/final_rerun3.log
: > $LOG
log() { echo "[$(date -Is)] $*" | tee -a $LOG; }

if ! bash -n $D/scripts/run_curves.sh; then
  log "FATAL: run_curves.sh fails bash -n"
  echo "FINAL_RERUN3_ABORT $(date -Is)" | tee -a $LOG
  exit 2
fi
log "run_curves.sh syntax OK sha256=$(sha256sum $D/scripts/run_curves.sh | cut -d' ' -f1)"

log "waiting for FINAL_RERUN2_DONE (profiles phase) before using GPU 2"
ABSENT=0
while true; do
  if grep -q FINAL_RERUN2_DONE $D/logs/final_rerun2.log 2>/dev/null; then
    log "rerun2 marker seen"
    break
  fi
  if pgrep -f 'run_profiles[.]sh' > /dev/null 2>&1; then
    ABSENT=0
  else
    ABSENT=$((ABSENT + 1))
    log "note: no run_profiles process (check $ABSENT/3)"
  fi
  if [ "$ABSENT" -ge 3 ]; then
    log "proceeding without rerun2 marker (no profiles process for 3 checks)"
    break
  fi
  sleep 60
done

PAIRS=""
for M in FasterGS FasterGSFused FastGS SADGS; do
  for SC in room bicycle garden; do
    [ -f $D/runs/$M/$SC/P16_RUN.json ] && PAIRS="$PAIRS $M:$SC"
  done
done
log "curve pairs:$PAIRS"
bash $D/scripts/run_curves.sh $PAIRS >> $LOG 2>&1
log "curves exit=$?"

echo "FINAL_RERUN3_DONE $(date -Is)" | tee -a $LOG
