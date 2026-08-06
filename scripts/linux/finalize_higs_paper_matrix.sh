#!/usr/bin/env bash
set -euo pipefail
WORKTREE="${WORKTREE:-/root/higs-paper-worktree}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/codex-3dgs-epic05/datasets/raw}"
PYTHON="${PYTHON:-/root/miniforge3/envs/gsplat/bin/python}"
ORIG_PYTHON="${ORIG_PYTHON:-/root/miniforge3/envs/train_original/bin/python}"
cd "$WORKTREE"

# 1. Reassemble any needs_assembly jobs from the 144-job matrix (idempotent).
"$PYTHON" src/scripts/run_higs_paper_a100_matrix.py \
  --root "$WORKTREE" \
  --data-root "$DATA_ROOT" \
  --source-higs artifacts/renderer-sources/gsplat-higs-v5 \
  --source-gsplat artifacts/renderer-sources/gsplat-official \
  --run-root artifacts/training-paper/runs \
  --result-root artifacts/training-paper/results \
  --session artifacts/training-paper/session.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --resume

# 2. Validate the full executable gsplat/HiGS subset (144 jobs).
"$PYTHON" src/scripts/validate_higs_paper_results.py \
  artifacts/training-paper/results/*.json \
  --methods gsplat,higs_full,higs_proposed --hardware a100 --require-complete

# 3. Original 3DGS smoke on one GPU (700 steps; non-paper-eligible).
SMOKE_SESSION=artifacts/training-original/smoke-session.json
"$ORIG_PYTHON" src/scripts/run_original_3dgs_a100_matrix.py \
  --root "$WORKTREE" \
  --data-root "$DATA_ROOT" \
  --source-original /root/renderer-candidates/original_3dgs_train \
  --run-root artifacts/training-original/smoke-runs \
  --result-root artifacts/training-original/smoke-results \
  --session "$SMOKE_SESSION" \
  --gpus 0 --max-jobs 1 --smoke-steps 700 \
  --resume
"$PYTHON" - "$SMOKE_SESSION" <<PYEOF
import json, sys
session = json.load(open(sys.argv[1]))
records = [r for r in session["jobs"].values() if r.get("status") == "smoke_complete"]
assert records, "original_3dgs smoke did not complete"
print("SMOKE_OK", records[0]["job_id"])
PYEOF

# 4. Launch the 33-job original_3dgs matrix on all 8 GPUs.
nohup bash scripts/linux/run_original_3dgs_a100_matrix.sh \
  > artifacts/training-original/matrix.log 2>&1 &
echo "original matrix launched, log: artifacts/training-original/matrix.log"
