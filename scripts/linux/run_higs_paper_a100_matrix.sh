#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${HIGS_PYTHON:-$HOME/miniforge3/envs/gsplat/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/codex-3dgs-epic05/datasets/raw}"
SOURCE_HIGS="${SOURCE_HIGS:-$ROOT/artifacts/renderer-sources/gsplat-higs-v5}"
SOURCE_GSPLAT="${SOURCE_GSPLAT:-$ROOT/artifacts/renderer-sources/gsplat-official}"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/training-paper/runs}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/artifacts/training-paper/results}"
SESSION="${SESSION:-$ROOT/artifacts/training-paper/session.json}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
METHODS="${METHODS:-gsplat,higs_full,higs_proposed}"
"$PYTHON" "$ROOT/src/scripts/run_higs_paper_a100_matrix.py" \
  --root "$ROOT" \
  --data-root "$DATA_ROOT" \
  --source-higs "$SOURCE_HIGS" \
  --source-gsplat "$SOURCE_GSPLAT" \
  --run-root "$RUN_ROOT" \
  --result-root "$RESULT_ROOT" \
  --session "$SESSION" \
  --gpus "$GPUS" \
  --methods "$METHODS" \
  --resume \
  "$@"
