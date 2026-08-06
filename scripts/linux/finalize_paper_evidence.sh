#!/usr/bin/env bash
set -euo pipefail
WORKTREE="${WORKTREE:-/root/higs-paper-worktree}"
PYTHON="${PYTHON:-/root/miniforge3/envs/gsplat/bin/python}"
cd "$WORKTREE"
echo "=== 1. validate original_3dgs subset (33 jobs) ==="
"$PYTHON" src/scripts/validate_higs_paper_results.py \
  artifacts/training-original/results/*.json \
  --methods original_3dgs --hardware a100 --require-complete
echo "=== 2. validate full 177-job evidence matrix ==="
"$PYTHON" src/scripts/validate_higs_paper_results.py \
  artifacts/training-paper/results/*.json artifacts/training-original/results/*.json \
  --require-complete
echo "=== 3. stage combined 177-job results dir ==="
rm -rf artifacts/training-all
mkdir -p artifacts/training-all/results
ln -sf ../../training-paper/results/*.json artifacts/training-all/results/
ln -sf ../../training-original/results/*.json artifacts/training-all/results/
ls artifacts/training-all/results/*.json | wc -l
echo "=== 4. build paper tables from 177 jobs ==="
"$PYTHON" src/scripts/build_higs_paper_tables.py \
  --results-dir artifacts/training-all/results --require-complete
echo "=== 5. show outputs ==="
ls -la paper/higs/tables/
echo "FINALIZE_OK"
