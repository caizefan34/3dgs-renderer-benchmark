#!/bin/bash
# P16-RB final phases RERUN 2 (after the opacity-reset dip fix, 2026-09-28):
#   - curves: nerficg pairs now run from the patched copy tree $D/curve_src/nerficg
#     (curve_schedule.patch) with P16_BACKUP_ITERS driving the checkpoint schedule at
#     offset -30 iterations (snapshots just before each opacity reset); graphdeco pairs
#     unchanged (their saves precede the in-loop reset by code ordering).
#   - profiles: unchanged from rerun 1 (harness bugs already fixed there).
# Old dip materials preserved by rename at curves/FasterGS/room_materials_dip_20260928
# and backups/dip_evidence_20260928/.
set -u
D=/root/epic05_p16rb
LOG=$D/logs/final_rerun2.log
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

echo "FINAL_RERUN2_DONE $(date -Is)" | tee -a $LOG
