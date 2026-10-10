#!/bin/bash
set -e
ZIP=$(find /mnt/storage_pool/liaoyuanjun /home/liaoyuanjun /tmp -name HIGS_R16_OPACITY_SEMANTICS_MX.zip -type f -print -quit 2>/dev/null)
test -n "$ZIP" || { echo ZIP_NOT_FOUND; exit 1; }
python3 -m zipfile -e "$ZIP" /mnt/storage_pool/liaoyuanjun
cd /mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX
source /home/liaoyuanjun/miniforge3/bin/activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env
CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh
