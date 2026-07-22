#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MINIFORGE_HOME="${MINIFORGE_HOME:-$HOME/miniforge3}"
ENV_ROOT="$MINIFORGE_HOME/envs"
GPU="${GPU:-7}"
PYTHON="$ENV_ROOT/gsplat/bin/python"

required_files=(
  "$ENV_ROOT/original3dgs/bin/python"
  "$ENV_ROOT/gsplat/bin/python"
  "$ENV_ROOT/speedy/bin/python"
  "$ENV_ROOT/tcgs/bin/python"
  "$ROOT/datasets/processed/mipnerf360/garden/point_cloud.ply"
  "$ROOT/datasets/processed/tanks_and_temples/truck/point_cloud.ply"
  "$ROOT/datasets/processed/tanks_and_temples/train/point_cloud.ply"
  "$ROOT/datasets/processed/mipnerf360/bicycle/point_cloud.ply"
  "$ROOT/datasets/processed/mipnerf360/bonsai/point_cloud.ply"
)
while true; do
  missing=0
  for path in "${required_files[@]}"; do
    [[ -f "$path" ]] || missing=$((missing + 1))
  done
  [[ "$missing" -eq 0 ]] && break
  echo "waiting for environment/data prerequisites: $missing missing"
  sleep 30
done

renderer_session="$ROOT/artifacts/run-logs/linux-all-configs-session.json"
renderer_resume=()
[[ -f "$renderer_session" ]] && renderer_resume=(--resume)
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/src/scripts/run_linux_tier_a_matrix.py" \
  --root "$ROOT" --env-root "$ENV_ROOT" --profile all-configs \
  --session "$renderer_session" --report-output "$ROOT/reports/generated/all-configs" \
  --wait-gpu "$GPU" --idle-max-memory-mib 1024 --idle-max-utilization 5 \
  --idle-samples 3 --idle-poll-seconds 30 "${renderer_resume[@]}"

compression_session="$ROOT/artifacts/run-logs/linux-compression-session.json"
compression_resume=()
[[ -f "$compression_session" ]] && compression_resume=(--resume)
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/src/scripts/run_linux_compression_matrix.py" \
  --root "$ROOT" --python "$PYTHON" --session "$compression_session" \
  --report-output "$ROOT/reports/generated/compression" --wait-gpu "$GPU" \
  --idle-max-memory-mib 1024 --idle-max-utilization 5 --idle-samples 3 \
  --idle-poll-seconds 30 "${compression_resume[@]}"

"$PYTHON" "$ROOT/src/scripts/run_temporal_matrix.py" \
  --source-root "$ROOT" --session "$renderer_session" \
  --output-root "$ROOT/results/measured-temporal" \
  --report-output "$ROOT/reports/generated/temporal"

MINIFORGE_HOME="$MINIFORGE_HOME" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
  CANDIDATE_ROOT="${CANDIDATE_ROOT:-$HOME/renderer-candidates}" \
  bash "$ROOT/scripts/linux/setup_candidate_envs.sh"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/src/scripts/run_candidate_smoke_matrix.py" \
  --root "$ROOT" --env-root "$ENV_ROOT" --wait-gpu "$GPU"

echo "EPIC-05 renderer, compression, temporal, and candidate pipeline complete"
