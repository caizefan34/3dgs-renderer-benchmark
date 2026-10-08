#!/bin/bash
# P16-RB final phases RERUN after fixing two harness bugs found in the first attempt:
#   - run_profiles.sh: `local M=$1 LOW=$2 ABI=$PROF/_ab/$M` expands $M before assignment
#     (dies under `set -u`); split into separate local statements.
#   - p16_curve_prepare.py: BACKUP.INTERVAL lives at TRAINING.BACKUP.INTERVAL (KeyError
#     'BACKUP' before) and `import yaml` is now lazy so the graphdeco pair prepare runs
#     under the gsplat env (no PyYAML there).
# Same order and GPU serialization as chain_final_phases.sh; stderr is captured this time.
# Evidence of the failed first attempt is preserved by rename under
# backups/final_phases_failed_run_20260928/.
set -u
D=/root/epic05_p16rb
LOG=$D/logs/final_rerun.log
: > $LOG
log() { echo "[$(date -Is)] $*" | tee -a $LOG; }

PAIRS=""
for M in FasterGS FasterGSFused FastGS SADGS; do
  for SC in room bicycle garden; do
    [ -f $D/runs/$M/$SC/P16_RUN.json ] && PAIRS="$PAIRS $M:$SC"
  done
done
log "curve pairs:$PAIRS"
bash $D/scripts/run_curves.sh $PAIRS >> $LOG 2>&1
log "curves exit=$?"

bash $D/scripts/run_profiles.sh >> $LOG 2>&1
log "profiles exit=$?"

echo "FINAL_RERUN_DONE $(date -Is)" | tee -a $LOG
