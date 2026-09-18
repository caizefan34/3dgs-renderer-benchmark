#!/bin/bash
# Build wrapper using CUDA 12.8 from higs-13scene-env (complete installation)
# but with conda env include paths for nv/target headers
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export PYTHONNOUSERSITE=1
cd "$HOME/3dgs-renderer-benchmark"
"$HOME/miniforge3/envs/anysplat/bin/python" experiments/r4/build_cuda_extension.py 2>&1
