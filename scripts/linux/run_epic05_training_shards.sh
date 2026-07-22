#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-$HOME/miniforge3/envs}"
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$HOME/renderer-candidates}"
PYTHON="$ENV_ROOT/gsplat/bin/python"
export EPIC05_TRAINING_COHORT_MODE="concurrent_8_gpu_sharded"

while true; do
  ready="$($PYTHON - "$ROOT" <<'PY' 2>/dev/null || true
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
checks = [
    (root / "artifacts/run-logs/linux-all-configs-session.json", "complete"),
    (root / "artifacts/run-logs/linux-compression-session.json", "complete"),
]
ok = all(path.is_file() and json.load(path.open()).get("status") == status for path, status in checks)
ok = ok and (root / "artifacts/candidate-smoke/summary.json").is_file()
print("yes" if ok else "no")
PY
)"
  [[ "$ready" == "yes" ]] && break
  echo "waiting for renderer/compression/candidate pipeline"
  sleep 60
done

for environment in train_original train_localgs train_gemmgs; do
  while [[ ! -x "$ENV_ROOT/$environment/bin/python" ]]; do
    echo "waiting for training environment: $environment"
    sleep 30
  done
done

run_shard() {
  local gpu="$1" backend="$2" session="$3"
  shift 3
  local case_args=()
  for case_id in "$@"; do
    case_args+=(--case "$case_id")
  done
  local resume=()
  [[ -f "$session" ]] && resume=(--resume)
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/src/scripts/run_linux_training_matrix.py" \
    --root "$ROOT" --repository-root "$REPOSITORY_ROOT" --env-root "$ENV_ROOT" \
    --output-root "$ROOT/artifacts/training" --session "$session" \
    --evaluator-python "$PYTHON" --backend "$backend" "${case_args[@]}" \
    --wait-gpu "$gpu" --idle-max-memory-mib 1024 --idle-max-utilization 5 \
    --idle-samples 3 --idle-poll-seconds 30 "${resume[@]}"
}

mkdir -p "$ROOT/artifacts/run-logs/training-shards"
run_shard 0 original_3dgs_train "$ROOT/artifacts/run-logs/training-shards/gpu0.json" \
  small-garden-1080p medium-truck-1080p &
run_shard 1 original_3dgs_train "$ROOT/artifacts/run-logs/training-shards/gpu1.json" \
  medium-train-1080p large-bicycle-1080p &
run_shard 2 original_3dgs_train "$ROOT/artifacts/run-logs/training-shards/gpu2.json" \
  large-bonsai-1080p &
run_shard 3 local_gs_train "$ROOT/artifacts/run-logs/training-shards/gpu3.json" \
  small-garden-1080p medium-truck-1080p &
run_shard 4 local_gs_train "$ROOT/artifacts/run-logs/training-shards/gpu4.json" \
  medium-train-1080p large-bicycle-1080p &
run_shard 5 local_gs_train "$ROOT/artifacts/run-logs/training-shards/gpu5.json" \
  large-bonsai-1080p &
run_shard 6 gemm_gs_train "$ROOT/artifacts/run-logs/training-shards/gpu6.json" \
  small-garden-1080p medium-truck-1080p medium-train-1080p &
run_shard 7 gemm_gs_train "$ROOT/artifacts/run-logs/training-shards/gpu7.json" \
  large-bicycle-1080p large-bonsai-1080p &
wait

"$PYTHON" "$ROOT/src/scripts/generate_training_report.py" --root "$ROOT"
echo "EPIC-05 native training shards complete"
