set -eu
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export TMPDIR=/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/tmp
export XDG_CACHE_HOME=/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/cache
export TORCH_EXTENSIONS_DIR=/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/torch_extensions
/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python /mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/commands/03_cpu_oracle_audit.py
