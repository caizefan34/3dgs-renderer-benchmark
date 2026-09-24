#!/usr/bin/env bash
set -euo pipefail

# Baseline: omit CANDIDATE_ROOT. Candidate: set it to the parent containing gsplat/.
EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR to experiments/final_sprint/warp_emit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$EXPERIMENT_DIR"
if [ -n "${CANDIDATE_ROOT:-}" ]; then
  export PYTHONPATH="$CANDIDATE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:?set candidate TORCH_EXTENSIONS_DIR}"
fi
"$PYTHON_BIN" warp_emit_pipeline.py
