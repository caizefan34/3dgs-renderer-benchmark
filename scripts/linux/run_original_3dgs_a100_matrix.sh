#!/usr/bin/env bash
set -euo pipefail
WORKTREE="${WORKTREE:-/root/higs-paper-worktree}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/codex-3dgs-epic05/datasets/raw}"
SOURCE_ORIGINAL="${SOURCE_ORIGINAL:-/root/renderer-candidates/original_3dgs_train}"
PYTHON="${PYTHON:-/root/miniforge3/envs/train_original/bin/python}"
cd "$WORKTREE"
exec "$PYTHON" src/scripts/run_original_3dgs_a100_matrix.py \
  --root "$WORKTREE" \
  --data-root "$DATA_ROOT" \
  --source-original "$SOURCE_ORIGINAL" \
  --run-root artifacts/training-original/runs \
  --result-root artifacts/training-original/results \
  --session artifacts/training-original/session.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --resume
