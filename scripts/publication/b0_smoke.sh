#!/bin/bash
# B0 trainer-integration smoke: 200 iterations, room, FUNCTIONAL_ONLY on a shared GPU.
# Exercises the EXACT trainer path: render -> loss -> backward -> densification stats
# (iteration 1!) -> clone/split/prune at 600 -> optimizer. Passes if it reaches iter 200.
export CUDA_VISIBLE_DEVICES=4
cd /home/liaoyuanjun/3dgs-renderer-benchmark
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python publication_trainer.py \
  --arm b0 --scene room --iterations 200 \
  --outdir /mnt/storage_pool/liaoyuanjun/pubphase/b0_smoke_200 \
  --gpu 0 --final-eval subset --timing-grade FUNCTIONAL_ONLY \
  > /mnt/storage_pool/liaoyuanjun/pubphase/b0_smoke.log 2>&1
echo "SMOKE_RC=$?" >> /mnt/storage_pool/liaoyuanjun/pubphase/b0_smoke.log
