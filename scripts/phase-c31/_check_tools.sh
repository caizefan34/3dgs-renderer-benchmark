#!/usr/bin/env bash
set -e
echo "=== Profiling tools on mx ==="
echo "nsys:"; which nsys 2>/dev/null || echo "  NOT FOUND"
echo "ncu:"; which ncu 2>/dev/null || echo "  NOT FOUND"
echo "nsight-sys:"; which nsight-sys 2>/dev/null || echo "  NOT FOUND"
echo "Python torch:" 
python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
echo "pip nsight:"; pip3 list 2>/dev/null | grep -i nsight || echo "  NONE"
echo "apt nsight:"; apt list --installed 2>/dev/null | grep nsight || echo "  NONE"
echo "nvidia tools:"; ls /usr/local/cuda/bin/nsys* 2>/dev/null || echo "  NO nsys in CUDA bin"
echo "cuda toolkit:"; cat /usr/local/cuda/version.txt 2>/dev/null || echo "  NO version.txt"
echo "nvcc:"; nvcc --version 2>/dev/null || echo "  NOT FOUND"
echo "torch profiler available:"
python3 -c 'import torch.profiler; print("YES")' 2>/dev/null || echo "NO"
