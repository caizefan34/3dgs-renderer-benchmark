#!/bin/bash
cd /mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX
source /home/liaoyuanjun/miniforge3/bin/activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx
python3 oracle_higs.py --capture captured_projected_inputs.pt --source-root /home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx --out-dir oracle
