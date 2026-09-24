#!/bin/bash
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export TORCH_CUDA_ARCH_LIST=8.0
export NVCC_FLAGS="-Xptxas -v"
export TORCH_EXTENSIONS_DIR=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_cache
export TMPDIR=/mnt/storage_pool/liaoyuanjun/tmp_p2_1a
echo "### debug: TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python -c "import torch,os; from torch.utils.cpp_extension import _get_build_directory; print('JIT_DIR=',_get_build_directory('experimental_gaussian_render_inference_scene_cuda', verbose=False)); print('env=',os.environ.get('TORCH_EXTENSIONS_DIR'))"