#!/bin/bash
# Correct driver: invoke the build function inside build.py.
set -u
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
CACHE=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_cache
LOG=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_build_fp32.log

export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export TORCH_CUDA_ARCH_LIST=8.0
export NVCC_FLAGS="-Xptxas -v"
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=/mnt/storage_pool/liaoyuanjun/tmp_p2_1a
export VERBOSE=1

rm -rf $CACHE
mkdir -p $CACHE /mnt/storage_pool/liaoyuanjun/tmp_p2_1a

cd /mnt/storage_pool/liaoyuanjun/higs_p2_1a_worktree/gsplat/experimental/render/kernels/cuda
echo "PWD=$(pwd)"
$PY -u -c "
import build
m = build.build_and_load_experimental_gaussian_render_inference_scene()
print('MODULE_LOADED', m.__file__)
" > $LOG 2>&1
RC=$?
echo "PY_RC=$RC"
echo "=== LOG WIDTH/LINES ==="
wc -lc < $LOG
echo "=== CACHE FIND ==="
find $CACHE -maxdepth 2 | head -40
echo "=== SO FILE ==="
find $CACHE -name '*.so' -exec ls -la {} \; -exec sha256sum {} \;
echo "=== TAIL LOG ==="
tail -50 $LOG
echo "BUILD4_DONE rc=$RC"