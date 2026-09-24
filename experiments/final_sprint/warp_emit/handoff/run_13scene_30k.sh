#!/usr/bin/env bash
set -euo pipefail

# DSH 13-scene launcher template. Keep baseline/candidate inputs, checkpoint,
# camera, seed, resolution, and repetition protocol identical. Do not compose
# Candidate C or any other feature with this run.
RUNNER="${WARP_EMIT_13_RUNNER:?set to the 13-scene checkpoint timing runner}"
MODE="${1:?usage: run_13scene_30k.sh baseline|candidate}"
SCENES="bicycle flowers garden stump treehill room counter kitchen bonsai train truck drjohnson playroom"
COMMON="--scenes $SCENES --checkpoint-tag 30k --tile-size 16 --warmup 20 --reps 40 --camera-index 0"
if [ "$MODE" = baseline ]; then
  exec "$RUNNER" $COMMON --variant baseline
elif [ "$MODE" = candidate ]; then
  : "${CANDIDATE_ROOT:?set CANDIDATE_ROOT}"
  export PYTHONPATH="$CANDIDATE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:?set candidate TORCH_EXTENSIONS_DIR}"
  exec "$RUNNER" $COMMON --variant warp_emit
else
  echo "mode must be baseline or candidate" >&2; exit 2
fi
