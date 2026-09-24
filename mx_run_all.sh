#!/bin/bash
set -e
cd /tmp/gsplat_systems_revalidation

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_systems
export CUDA_CACHE_PATH=/tmp/cuda_cache_systems
export CUDA_VISIBLE_DEVICES=1

# Now run the main script
cd /tmp/gsplat_systems_revalidation
python3 tmp_systems_validation_v2.py --stage all 2>&1
echo "EXIT CODE: $?"
