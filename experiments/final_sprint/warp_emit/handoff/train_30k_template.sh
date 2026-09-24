#!/usr/bin/env bash
set -euo pipefail

# Reference V1 trainer template. Use distinct output dirs and the same seed,
# camera sequence, data root, and optimizer settings for baseline/candidate.
REPO_ROOT="${REPO_ROOT:?set repository root}"
SCENE="${1:?usage: train_30k_template.sh scene baseline|candidate}"
MODE="${2:?usage: train_30k_template.sh scene baseline|candidate}"
OUT_DIR="${OUT_DIR:?set output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ "$MODE" = candidate ]; then
  : "${CANDIDATE_ROOT:?set CANDIDATE_ROOT}"
  export PYTHONPATH="$CANDIDATE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:?set candidate TORCH_EXTENSIONS_DIR}"
elif [ "$MODE" != baseline ]; then
  echo "mode must be baseline or candidate" >&2; exit 2
fi
exec "$PYTHON_BIN" "$REPO_ROOT/baseline/reference_v1/trainer.py" \
  --scene "$SCENE" --iterations 30000 --output_dir "$OUT_DIR" --allow_dirty
