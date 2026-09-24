#!/bin/bash
# Launch P3-H build for the given variant(s). Usage: _build.sh V0 | --all
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
cd /tmp
nohup /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python \
  /tmp/p3_h_accum_ceiling.py "$@" \
  > /tmp/p3h_build.log 2>&1 &
echo "PID=$!"