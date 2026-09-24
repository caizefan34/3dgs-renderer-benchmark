#!/bin/bash
set -x
ls -la /mnt/storage_pool/liaoyuanjun/pubphase/b0_build.log 2>&1
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/usr/local/cuda/bin:$PATH
which ninja nvcc git
echo "== github reachability =="
timeout 60 git ls-remote https://github.com/graphdeco-inria/gaussian-splatting HEAD 2>&1 | head -3
echo "== clone attempt =="
if [ ! -d /mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub ]; then
  timeout 300 git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting /mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub 2>&1 | tail -6
fi
ls /mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub 2>/dev/null | head
echo DIAG_DONE
