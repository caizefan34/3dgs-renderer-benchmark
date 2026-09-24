#!/usr/bin/env bash
set -euo pipefail

# Run from the experiment parent that contains warp_emit_correctness.py.
EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR to experiments/final_sprint/warp_emit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$EXPERIMENT_DIR"
"$PYTHON_BIN" warp_emit_correctness.py baseline
PYTHONPATH="${CANDIDATE_ROOT:?set CANDIDATE_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:?set TORCH_EXTENSIONS_DIR}" \
"$PYTHON_BIN" warp_emit_correctness.py candidate
