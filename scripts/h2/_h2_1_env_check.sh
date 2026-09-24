#!/bin/bash
# H2-1 environment verification for mx
set -e
echo "=== ENV CHECK ==="
source /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11', torch._C._GLIBCXX_USE_CXX11_ABI)"

echo "=== GSPLAT CHECK ==="
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda
export CUDA_HOME=$CONDA_PREFIX
python -c "from gsplat.experimental import rasterize_gaussian_higs_frozen; print('B2 import OK')"

echo "=== PLY CHECK ==="
ls -la /mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply

echo "=== CAMS CHECK ==="
ls -la /mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json

echo "=== SCRIPT CHECK ==="
ls -la /tmp/h2_1_profile.py

echo "=== GPU CHECK ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "=== ALL CHECKS PASSED ==="
