#!/usr/bin/env bash
set -euo pipefail

root="${1:-/root/codex-3dgs-full-optimization}"
python_bin="${2:-/root/miniforge3/envs/gsplat/bin/python}"
decoded_root="${3:-/root/codex-fcgs-rate-curve}"
output_root="${4:-$root/artifacts/compression/runs/fcgs-rate-curve-medium-train-1080p}"
gpu="${CUDA_VISIBLE_DEVICES:-2}"

cd "$root"
for lmd in 0.0001 0.0002 0.0004 0.0008 0.0016; do
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" src/scripts/run_local_renderer_suite.py \
    --scene "$decoded_root/train-lmd-$lmd.ply" \
    --cameras datasets/processed/tanks_and_temples/train/eval_cameras.json \
    --ground-truth-dir datasets/processed/tanks_and_temples/train/eval_images \
    --renderers gsplat \
    --output-dir "$output_root/lmd-$lmd" \
    --frames 100 \
    --warmup 30 \
    --repeats 5 \
    --width 1920 \
    --height 1080
done
