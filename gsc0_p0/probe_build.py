export PYTHONIOENCODING=utf-8
echo "===== nvcc / CUDA ====="
which nvcc && nvcc --version 2>&1 | tail -3
echo "CUDA_HOME=$CUDA_HOME"
ls -d /usr/local/cuda* 2>/dev/null
echo "===== true_accutile csrc files ====="
ls /root/epic05_frozen/gsplat-true-accutile-v153/gsplat/cuda/csrc/*.cu 2>/dev/null | wc -l
ls /root/epic05_frozen/gsplat-true-accutile-v153/gsplat/cuda/ext.cpp 2>/dev/null
echo "===== versioner update_hash (crash site) ====="
sed -n "1,30p" /root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/utils/_cpp_extension_versioner.py
echo "===== ninja available? ====="
/root/miniforge3/envs/gsplat/bin/python -c "import ninja; print('ninja', ninja.__version__)" 2>&1
ninja --version 2>&1