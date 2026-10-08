#!/bin/bash
# P16-RB final phases: wait for the stage-2 chain to finish, then wait for the
# operator review gate file, then run step 10 (TTQ curves) and step 11 (bottleneck
# profiling replays) sequentially on the single physical GPU 2.
set -u
D=/root/epic05_p16rb
LOG=$D/logs/chain_final.log
: > $LOG
log() { echo "[$(date -Is)] $*" | tee -a $LOG; }

log "chain_final start; waiting for CHAIN_STAGE2_DONE"
until grep -q "CHAIN_STAGE2_DONE" $D/logs/chain_stage2.log 2>/dev/null; do sleep 60; done
log "stage2 chain done"

log "operator review point: waiting for $D/GATE_CLEARED"
until [ -f $D/GATE_CLEARED ]; do sleep 60; done
log "gate cleared: $(cat $D/GATE_CLEARED)"

PAIRS=""
for M in FasterGS FasterGSFused FastGS SADGS; do
  for SC in room bicycle garden; do
    [ -f $D/runs/$M/$SC/P16_RUN.json ] && PAIRS="$PAIRS $M:$SC"
  done
done
log "curve pairs:$PAIRS"
bash $D/scripts/run_curves.sh $PAIRS >> $LOG 2>&1
log "curves exit=$?"

bash $D/scripts/run_profiles.sh >> $LOG
log "profiles exit=$?"

echo "CHAIN_FINAL_DONE $(date -Is)" | tee -a $LOG
