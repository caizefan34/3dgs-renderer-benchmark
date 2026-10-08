#!/bin/bash
# N1 Stage 0 remote environment audit
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
echo "=== python/torch ==="
$PY -c "import sys, torch; print(sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "=== analysis libs ==="
$PY -c "import sklearn; print('sklearn', sklearn.__version__)" 2>&1 | tail -1
$PY -c "import scipy; print('scipy', scipy.__version__)" 2>&1 | tail -1
$PY -c "import pyarrow; print('pyarrow', pyarrow.__version__)" 2>&1 | tail -1
$PY -c "import pandas; print('pandas', pandas.__version__)" 2>&1 | tail -1
$PY -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | tail -1
echo "=== disk ==="
df -h /root | tail -1
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== wrapper ==="
cat /root/epic05_py_wrapper.sh
echo "=== frozen hashes (Stage 0) ==="
sha256sum /root/epic05_p12/publication_trainer_p12.py
sha256sum /root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
sha256sum /root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so
sha256sum /root/fused-ssim/fused_ssim_cuda/fused_ssim_cuda.so 2>/dev/null || find /root -name "fused_ssim_cuda.so" 2>/dev/null | head -3
echo "=== reference_v1 files ==="
ls /root/3dgs-renderer-benchmark/baseline/reference_v1/
echo "=== dataset cams ==="
ls /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/ | head
ls /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/bicycle/ | head
echo "=== n1 dir ==="
ls -la /root/epic05_n1 2>/dev/null || echo "epic05_n1 does not exist yet"
