#!/bin/bash
# P16-RB: unattended chain — wait for Stage 1 -> finalize room runs -> room gate ->
# Stage 2 (bicycle+garden) for passing methods -> finalize Stage 2 records.
# Deliberately stops before curves/profiles so the operator reviews room results first.
set -u
D=/root/epic05_p16rb
PY=/root/miniforge3/envs/p16_fastergs/bin/python
LOG=$D/logs/chain_stage2.log
: > $LOG
log() { echo "[$(date -Is)] $*" | tee -a $LOG; }

# 1. wait for Stage 1 completion (up to 20 h)
for i in $(seq 1 1200); do
  grep -q STAGE1_ALL_DONE $D/logs/stage1.log 2>/dev/null && break
  sleep 60
done
grep -q STAGE1_ALL_DONE $D/logs/stage1.log 2>/dev/null || { log "TIMEOUT waiting for stage1"; exit 1; }
log "stage1 complete; finalizing room runs"

# 2. finalize room (metrics on GPU 3)
bash $D/scripts/finalize_matrix.sh stage1 FasterGS:room FasterGSFused:room FastGS:room SADGS:room | tee -a $LOG

# 3. room gate -> expansion list
$PY $D/scripts/p16_room_gate.py --out $D/artifacts/P16_ROOM_GATE.json 2>&1 | tee -a $LOG
PASS=$($PY -c "import json;print(' '.join(json.load(open('$D/artifacts/P16_ROOM_GATE.json'))['passing']))")
if [ -z "$PASS" ]; then
  log "no method passed the room gate; STAGE2_SKIPPED"
  echo "CHAIN_STAGE2_DONE_NO_PASS $(date -Is)" | tee -a $LOG
  exit 0
fi
log "room gate passing methods: $PASS"

# 4. stage 2
bash $D/scripts/run_stage2_triad.sh $PASS

# 5. finalize stage-2 runs
PAIRS=""
for M in $PASS; do PAIRS="$PAIRS $M:bicycle $M:garden"; done
bash $D/scripts/finalize_matrix.sh stage2 $PAIRS | tee -a $LOG
log "chain complete; curves/profiles not launched (operator review point)"
echo "CHAIN_STAGE2_DONE $(date -Is)" | tee -a $LOG
