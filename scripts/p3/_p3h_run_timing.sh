#!/bin/bash
# DSH-H timing run — idle GPU 0, full protocol (20w / 5r / 100s), all 5 variants, 3 scenes.
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export PYTHONPATH=/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint
export HIGS_BWD_H8_MR=1
export HIGS_PX_RUNTIME=2
cd /mnt/storage_pool/liaoyuanjun
python /mnt/storage_pool/liaoyuanjun/p3_h_run.py timing --all --scenes room,bicycle,garden --warmup 20 --reps 5 --samples 100 > /mnt/storage_pool/liaoyuanjun/p3h_timing.log 2>&1
echo "TIMING_DONE rc=$?"