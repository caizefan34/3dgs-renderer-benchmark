echo "=== nvcc ==="; which nvcc && nvcc --version 2>&1 | tail -3 || echo "NO nvcc in PATH"
echo "=== CUDA toolkit dirs ==="; ls -d /usr/local/cuda* 2>/dev/null; ls /usr/local/cuda/bin/nvcc 2>/dev/null && /usr/local/cuda/bin/nvcc --version 2>&1 | tail -3
echo "=== gcc/g++ ==="; gcc --version 2>&1 | head -1; g++ --version 2>&1 | head -1
echo "=== torch cuda ==="; /mnt/storage_pool/liaoyuanjun_mx0qual/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available());print('arch',torch.cuda.get_device_name(0))"
echo "=== cuda home env ==="; echo "CUDA_HOME=$CUDA_HOME"; echo "CUDA_PATH=$CUDA_PATH"
echo "=== nvidia-smi driver ==="; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
