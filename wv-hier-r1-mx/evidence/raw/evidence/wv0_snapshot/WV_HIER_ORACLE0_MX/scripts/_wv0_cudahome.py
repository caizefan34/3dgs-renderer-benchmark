import os, glob, subprocess
E="/home/liaoyuanjun/miniforge3/envs/anysplat"
print("nvcc", os.path.exists(E+"/bin/nvcc"))
print("include/cuda_runtime.h", os.path.exists(E+"/include/cuda_runtime.h"))
print("include dirs:", [d for d in glob.glob(E+"/include/*")][:5])
# libcudart symlink
print("lib64 libcudart", glob.glob(E+"/lib64/libcudart*") or glob.glob(E+"/lib/libcudart*"))
