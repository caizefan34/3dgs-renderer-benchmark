#!/bin/bash
# Wrapper to build CUDA extension with correct PATH
export PATH="$HOME/miniforge3/envs/anysplat/bin:$PATH"
export CUDA_HOME=/tmp/cuda_12_4
export PYTHONNOUSERSITE=1
cd "$HOME/3dgs-renderer-benchmark"
"$HOME/miniforge3/envs/anysplat/bin/python" experiments/r4/build_cuda_extension.py 2>&1
