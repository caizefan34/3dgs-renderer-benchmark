#!/bin/bash
# Forced-verbose diagnostic build under the frozen env.
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
export MAX_JOBS=1

echo "=== WHICH ==="
which python
$PY --version 2>&1
echo "=== TORCH ==="
$PY -c "import torch,torch.utils.cpp_extension as j;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('gethome',j.CUDA_HOME)" 2>&1
echo "=== NVCC ==="
ls -la /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/nvcc 2>&1
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/nvcc --version 2>&1 | head -5

echo "=== CLEAR CACHE ==="
rm -rf $CACHE
mkdir -p $CACHE /mnt/storage_pool/liaoyuanjun/tmp_p2_1a

cd /mnt/storage_pool/liaoyuanjun/higs_p2_1a_worktree/gsplat/experimental/render/kernels/cuda

echo "=== BUILD DIR CHECK ==="
$PY -c "
import sys,os
sys.argv=['x']
import build
b=build.get_build_parameters()
import torch.utils.cpp_extension as j
d=j._get_build_directory(b.name,verbose=False)
print('name=',b.name)
print('build_dir=',d)
print('expected=',os.path.join(os.environ['TORCH_EXTENSIONS_DIR'],'experimental_gaussian_render_inference_scene_cuda'))
" 2>&1

echo "=== RUNNING BUILD (VERBOSE=1, redirected) ==="
cd /mnt/storage_pool/liaoyuanjun/higs_p2_1a_worktree/gsplat/experimental/render/kernels/cuda
$PY -u build.py > $LOG 2>&1
RC=$?
echo "PY_RC=$RC"
echo "=== LOG SIZE ==="
wc -c < $LOG
echo "=== CACHE CONTENTS ==="
find $CACHE -maxdepth 2 | head -50
echo "=== TAIL LOG ==="
tail -40 $LOG
echo "DIAG2_DONE rc=$RC"