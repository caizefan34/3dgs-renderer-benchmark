#!/usr/bin/env bash
set -euo pipefail

# BASELINE_ROOT is the directory containing the untouched gsplat package.
BASELINE_ROOT="${BASELINE_ROOT:?set BASELINE_ROOT to a directory containing gsplat/}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/torch_extensions_baseline}"
"$PYTHON_BIN" -c 'from gsplat.cuda._backend import _C; assert _C is not None; print("baseline gsplat ready")'
