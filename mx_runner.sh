#!/bin/bash
# Runner script for mx server - DSH-2 Systems Revalidation
set -e

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_systems
export CUDA_CACHE_PATH=/tmp/cuda_cache_systems
export CUDA_VISIBLE_DEVICES=1

cd /tmp/gsplat_systems_revalidation

STAGE=${1:-all}

echo "=== Running Stage: $STAGE ==="
echo "GPU: $(nvidia-smi --id=1 --query-gpu=index,name --format=csv,noheader 2>/dev/null)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "gsplat: $(python3 -c 'import gsplat; print(gsplat.__version__)')"
echo ""

# Run validation
python3 tmp_systems_validation_v2.py --stage "$STAGE"

echo ""
echo "=== Results ==="
ls -la results/reference_v1/systems_revalidation/
