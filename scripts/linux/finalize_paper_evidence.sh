#!/usr/bin/env bash
set -euo pipefail
WORKTREE="${WORKTREE:-/root/higs-paper-worktree}"
PYTHON="${PYTHON:-/root/miniforge3/envs/gsplat/bin/python}"
cd "$WORKTREE"
echo "=== 1. validate original_3dgs subset (33 jobs) ==="
"$PYTHON" src/scripts/validate_higs_paper_results.py \
  artifacts/training-original/results/*.json \
  --methods original_3dgs --hardware a100 --require-complete
echo "=== 2. validate speedy_splat subset (33 jobs) ==="
"$PYTHON" src/scripts/validate_higs_paper_results.py \
  artifacts/training-speedy/results/*.json \
  --methods speedy_splat --hardware a100 --require-complete
echo "=== 3. validate full 210-job evidence matrix ==="
"$PYTHON" src/scripts/validate_higs_paper_results.py \
  artifacts/training-paper/results/*.json artifacts/training-original/results/*.json \
  artifacts/training-speedy/results/*.json \
  --methods original_3dgs,gsplat,higs_full,higs_proposed,speedy_splat --hardware a100 \
  --require-complete
echo "=== 4. stage combined 210-job results dir ==="
rm -rf artifacts/training-all
mkdir -p artifacts/training-all/results
cp artifacts/training-paper/results/*.json artifacts/training-all/results/
cp artifacts/training-original/results/*.json artifacts/training-all/results/
cp artifacts/training-speedy/results/*.json artifacts/training-all/results/
ls artifacts/training-all/results/*.json | wc -l
echo "=== 5. build paper tables from 210 jobs ==="
"$PYTHON" src/scripts/build_higs_paper_tables.py \
  --results-dir artifacts/training-all/results \
  --methods original_3dgs,gsplat,higs_full,higs_proposed,speedy_splat --hardware a100 \
  --require-complete
echo "=== 6. show outputs ==="
ls -la paper/higs/tables/
echo "FINALIZE_OK"
