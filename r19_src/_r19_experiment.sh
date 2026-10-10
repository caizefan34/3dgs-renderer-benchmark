#!/usr/bin/env bash
# R19 full experiment driver: P1 (done separately) + P2 5K pilot + P3 30K.
# Usage: _r19_experiment.sh <phase> [iters] [scenes...]
#   phase: p2 (5K pilot) | p3 (30K) | both
set -u
W=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_FULLTRAIN_R19_MX
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3
export HIGS_PROJECT=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx
export R19_OUT=$W
export PYTHONNOUSERSITE=1 TMPDIR=/mnt/storage_pool/liaoyuanjun/tmp
export CUDA_VISIBLE_DEVICES=0
export PYTHON=$PY
cd $W/scripts

PHASE="${1:-p2}"
ITERS="${2:-5000}"
SCENES=("${@:3}")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(bicycle garden room)
fi

run_one() {
  local be=$1 scene=$2 iters=$3 tag=$4
  local outdir=$W/raw/${tag}_${scene}
  echo ">>> [$tag] $be $scene iters=$iters"
  $PY r19_trainer.py --backend $be --scene $scene --iterations $iters \
      --outdir $outdir --lpips 2>&1 | grep -vE "Caching|Pre-caching|Loading cameras|matched|images cached" | tail -8
  echo ">>> [$tag] $be $scene EXIT=${PIPESTATUS[0]}"
}

if [ "$PHASE" = "p2" ] || [ "$PHASE" = "both" ]; then
  for s in "${SCENES[@]}"; do
    run_one std_ll $s $ITERS p2
    run_one higs   $s $ITERS p2
  done
fi
if [ "$PHASE" = "p3" ] || [ "$PHASE" = "both" ]; then
  for s in "${SCENES[@]}"; do
    run_one std_ll $s $ITERS p3
    run_one higs   $s $ITERS p3
  done
fi
echo "EXPERIMENT_DONE phase=$PHASE iters=$ITERS"
