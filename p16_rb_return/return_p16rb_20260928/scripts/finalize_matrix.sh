#!/bin/bash
# P16-RB: finalize a set of completed runs (name map -> common metrics -> P16_RUN.json).
# Metric evaluation uses GPU 3 so it can never contaminate the training card (GPU 2).
#
# usage: finalize_matrix.sh stage1 FasterGS:room ...   |   finalize_matrix.sh stage2 FastGS:bicycle ...
set -u
D=/root/epic05_p16rb
STAGE=$1; shift
PY=/root/miniforge3/envs/p16_fastergs/bin/python
LOG=$D/logs/finalize_${STAGE}.log
: > $LOG

commit_of() {
  case $1 in
    FasterGS) echo 117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847 ;;
    FasterGSFused) echo ae665af2f91a4973bb92c40fdc4cba8d12fe3c96 ;;
    FastGS) echo 44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f ;;
    SADGS) echo 029814fb5ed773e4a794cec5115951a64f610ab8 ;;
  esac
}
note_of() {
  case $1 in
    FasterGS) echo "nerficg harness e458ac6e (v2.2); author recipe fastergs_garden.yaml schema; IMAGE_SCALE_FACTOR 1.0 common-resolution" ;;
    FasterGSFused) echo "nerficg harness e458ac6e (v2.2); upstream fused branch; author recipe fastergsfused_garden.yaml schema; IMAGE_SCALE_FACTOR 1.0" ;;
    FastGS) echo "upstream train_base.sh scene args at pinned commit; -r 1 common-resolution; densification_interval 500" ;;
    SADGS) echo "upstream run_train.sh scene config at pinned commit; -r 1 common-resolution; mult 0.25; structure-aware densification" ;;
  esac
}
n_of() { case $1 in SADGS) echo 3000 ;; *) echo 30000 ;; esac; }

for PAIR in "$@"; do
  M=${PAIR%%:*}; SC=${PAIR##*:}
  if [ "$STAGE" = "stage1" ]; then
    VCSV=$D/logs/stage1_${M}.vram.csv
    SLOG=$D/logs/stage1_${M}_train.stdout.log
  else
    VCSV=$D/logs/stage2_${M}_${SC}.vram.csv
    SLOG=$D/logs/stage2_${M}_${SC}_train.stdout.log
  fi
  echo "[$(date -Is)] finalize $M $SC (stage=$STAGE)" | tee -a $LOG
  $PY $D/scripts/p16_finalize_run.py --method $M --scene $SC \
    --source-commit "$(commit_of $M)" --native-iterations "$(n_of $M)" \
    --vram-csv $VCSV --stdout-log $SLOG \
    --recipe-note "$(note_of $M)" --eval-gpu 3 --target-gpu 2 >> $LOG 2>&1 \
    && echo "FINALIZED_${M}_${SC}" | tee -a $LOG \
    || echo "FINALIZE_FAIL_${M}_${SC}" | tee -a $LOG
done
