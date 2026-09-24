#!/bin/bash
# Run smoke on all variants.
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
nohup /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python \
  /tmp/p3_h_accum_ceiling.py smoke \
  > /tmp/p3h_smoke.log 2>&1 &
echo "PID=$!"