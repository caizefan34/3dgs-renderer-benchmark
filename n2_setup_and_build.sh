#!/bin/bash
set -e

# ====================================================================
# N2 CPCB â€?Setup, Provenance, Build BASELINE and CPCB
# Uses miniforge3 CUDA 11.8 toolkit (matches torch 2.7.1+cu118)
# ====================================================================

WORK="/tmp/gsplat_n2_cpcb"
LEGO_SRC="$HOME/projects/LEGO/third_party/gsplat"
RESULTS_DIR="$HOME/3dgs-renderer-benchmark/results/n2_cpcb"
BUILD_LOGS="$RESULTS_DIR/build_logs"

# Use miniforge3 base CUDA 11.8 toolkit (complete: nvcc + cicc + headers + libdevice)
# But keep system python3 (which has torch) as primary python.
export CUDA_HOME="$HOME/miniforge3"
# Only prepend nvcc tools, NOT python3 â€?system python3 has torch/gsplat
export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"
# Ensure system python3 stays primary
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
# Explicitly set which python to use (system python3 with torch 2.7.1+cu118)
PYBIN="/usr/bin/python3"

mkdir -p "$RESULTS_DIR" "$BUILD_LOGS"

echo "============================================================"
echo "N2 CPCB SETUP â€?$(date)"
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $(nvcc --version 2>&1 | grep release)"
echo "============================================================"

# --- 0. Record provenance ---
echo "=== RECORDING PROVENANCE ==="
{
  echo "{"
  echo '  "timestamp": "'$(date -Iseconds)'",'
  echo '  "hostname": "'$(hostname)'",'
  echo -n '  "gpu": '
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1 | awk -F', ' '{printf "{\"name\":\"%s\",\"memory_mib\":%s,\"driver\":\"%s\"}",$1,$2,$3}'
  echo ","
  echo -n '  "nvcc": "'
  nvcc --version 2>/dev/null | grep "release" | sed 's/.*release /release /' | tr -d '\n'
  echo '",'
  echo -n '  "cuda_home": "'
  echo -n "$CUDA_HOME" | tr -d '\n'
  echo '",'
  echo -n '  "python": "'
  $PYBIN --version 2>&1 | tr -d '\n'
  echo '",'
  echo -n '  "torch": "'
  $PYBIN -c "import torch; print(torch.__version__)" 2>/dev/null | tr -d '\n'
  echo '",'
  echo -n '  "torch_cuda": "'
  $PYBIN -c "import torch; print(torch.version.cuda)" 2>/dev/null | tr -d '\n'
  echo '",'
  echo -n '  "gsplat_version": "'
  $PYBIN -c "import gsplat; print(gsplat.__version__)" 2>/dev/null | tr -d '\n'
  echo '"'
  echo "}"
} > "$RESULTS_DIR/provenance.json"
cat "$RESULTS_DIR/provenance.json"

# Record git state of canonical repo
cd "$HOME/3dgs-renderer-benchmark"
{
  echo "{"
  echo '  "git_head": "'$(git rev-parse HEAD)'",'
  echo '  "git_describe": "'$(git describe --tags --always 2>/dev/null)'",'
  echo -n '  "git_status_short": "'
  git status --short | head -5 | tr '\n' ';' | sed 's/"/'\''/g' | tr -d '\n'
  echo '"'
  echo "}"
} > "$RESULTS_DIR/git_state.json"

# Record sha256 of key files
echo "=== SHA256 OF KEY FILES ==="
for f in baseline/reference_v1/trainer.py baseline/reference_v1/config.py baseline/reference_v1/gaussian_model.py; do
  if [ -f "$f" ]; then
    echo "  $f: $(sha256sum "$f" | cut -d' ' -f1)"
  fi
done

INSTALLED_CSRC="$HOME/.local/lib/python3.10/site-packages/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu"
echo "  installed RasterizeToPixels3DGSBwd.cu: $(sha256sum "$INSTALLED_CSRC" | cut -d' ' -f1)"

# --- 1. Create isolated worktrees ---
echo "=== CREATING WORKTREES ==="
rm -rf "$WORK"
mkdir -p "$WORK"

cp -r "$LEGO_SRC" "$WORK/gsplat_baseline"
cp -r "$LEGO_SRC" "$WORK/gsplat_cpcb"

echo "Baseline source: $WORK/gsplat_baseline"
echo "CPCB source: $WORK/gsplat_cpcb"
echo "  LEGO source RasterizeToPixels3DGSBwd.cu: $(sha256sum "$WORK/gsplat_baseline/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu" | cut -d' ' -f1)"

# --- 2. Apply CPCB patch ---
echo "=== APPLYING CPCB PATCH ==="
cd "$WORK/gsplat_cpcb"
$PYBIN "$HOME/3dgs-renderer-benchmark/n2_apply_patch.py" "gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu"

diff -u "$WORK/gsplat_baseline/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu" \
        "$WORK/gsplat_cpcb/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu" > "$RESULTS_DIR/cpcb_diff.patch" || true
echo "Diff saved to $RESULTS_DIR/cpcb_diff.patch"
cat "$RESULTS_DIR/cpcb_diff.patch"
echo "  patched RasterizeToPixels3DGSBwd.cu: $(sha256sum "$WORK/gsplat_cpcb/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu" | cut -d' ' -f1)"

# --- 3. Build BASELINE ---
echo "=== BUILDING BASELINE ==="
cd "$WORK/gsplat_baseline"

export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=16
export TORCH_EXTENSIONS_DIR="$WORK/torch_ext_baseline"
# Add ptxas verbose for resource auditing
export NVCC_FLAGS="-Xptxas -v --use_fast_math -O3 -std=c++17 --expt-relaxed-constexpr"

echo "Building baseline with CUDA_HOME=$CUDA_HOME, TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
$PYBIN setup.py build_ext --inplace 2>&1 | tee "$BUILD_LOGS/baseline_build.log"

BASELINE_SO=$(find "$WORK/gsplat_baseline" -name "csrc*.so" | head -1)
echo "Baseline .so: $BASELINE_SO"
if [ -z "$BASELINE_SO" ]; then
    echo "ERROR: Baseline build failed - no .so found"
    exit 1
fi

# --- 4. Build CPCB ---
echo "=== BUILDING CPCB ==="
cd "$WORK/gsplat_cpcb"
export TORCH_EXTENSIONS_DIR="$WORK/torch_ext_cpcb"

echo "Building CPCB with CUDA_HOME=$CUDA_HOME, TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
$PYBIN setup.py build_ext --inplace 2>&1 | tee "$BUILD_LOGS/cpcb_build.log"

CPCB_SO=$(find "$WORK/gsplat_cpcb" -name "csrc*.so" | head -1)
echo "CPCB .so: $CPCB_SO"
if [ -z "$CPCB_SO" ]; then
    echo "ERROR: CPCB build failed - no .so found"
    exit 1
fi

# --- 5. Extract ptxas resource metrics ---
echo "=== EXTRACTING PTXAS METRICS ==="
$PYBIN "$HOME/3dgs-renderer-benchmark/n2_extract_ptxas.py" \
    "$BUILD_LOGS/baseline_build.log" "$BUILD_LOGS/cpcb_build.log" \
    "$RESULTS_DIR/compiler_resources.json" 2>&1 || echo "ptxas extraction warning"

# --- 6. Verify both builds load ---
echo "=== VERIFYING BUILDS ==="
PYTHONPATH="$WORK/gsplat_baseline" $PYBIN -c "
import gsplat; print('baseline gsplat:', gsplat.__version__, gsplat.__file__)
from gsplat.cuda._backend import _C; print('baseline _C:', type(_C))
" 2>&1 | tee -a "$BUILD_LOGS/baseline_build.log"

PYTHONPATH="$WORK/gsplat_cpcb" $PYBIN -c "
import gsplat; print('cpcb gsplat:', gsplat.__version__, gsplat.__file__)
from gsplat.cuda._backend import _C; print('cpcb _C:', type(_C))
" 2>&1 | tee -a "$BUILD_LOGS/cpcb_build.log"

echo "============================================================"
echo "N2 SETUP + BUILD COMPLETE"
echo "  Baseline: $WORK/gsplat_baseline ($BASELINE_SO)"
echo "  CPCB:     $WORK/gsplat_cpcb ($CPCB_SO)"
echo "  Results:  $RESULTS_DIR"
echo "============================================================"
