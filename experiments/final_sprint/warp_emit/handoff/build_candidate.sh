#!/usr/bin/env bash
set -euo pipefail

# Makes an isolated candidate copy, preserving the original package untouched.
HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSPAT_SOURCE="${GSPAT_SOURCE:?set GSPAT_SOURCE to untouched gsplat/}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:?set CANDIDATE_ROOT to an empty writable directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
test -d "$GSPAT_SOURCE/cuda/csrc"
test ! -e "$CANDIDATE_ROOT/gsplat" || { echo "candidate exists: $CANDIDATE_ROOT/gsplat" >&2; exit 2; }
mkdir -p "$CANDIDATE_ROOT"
cp -a "$GSPAT_SOURCE" "$CANDIDATE_ROOT/gsplat"
for binary in "$CANDIDATE_ROOT"/gsplat/csrc*.so; do
  [ -e "$binary" ] && mv "$binary" "$binary.baseline_binary"
done
(cd "$CANDIDATE_ROOT/gsplat" && patch -p1 < "$HANDOFF_DIR/warp_emit.patch")
export PYTHONPATH="$CANDIDATE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CANDIDATE_ROOT/torch_extensions}"
"$PYTHON_BIN" -c 'from gsplat.cuda._backend import _C; assert _C is not None; print("WARP_ALL candidate ready")'
