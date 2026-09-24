#!/usr/bin/env bash
set -euo pipefail

# The evaluator is project-owned; pass matched 30K baseline/candidate outputs.
EVAL_RUNNER="${WARP_EMIT_EVAL_RUNNER:?set evaluator executable}"
BASELINE_CKPT="${BASELINE_CKPT:?set baseline checkpoint}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:?set candidate checkpoint}"
exec "$EVAL_RUNNER" --baseline "$BASELINE_CKPT" --candidate "$CANDIDATE_CKPT" \
  --metrics rgb alpha psnr loss --same-cameras
