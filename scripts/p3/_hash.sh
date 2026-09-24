#!/bin/bash
# Run hash + emit-static on mx. Usage: _hash.sh
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python /tmp/p3_h_accum_ceiling.py hash
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python /tmp/p3_h_accum_ceiling.py emit-static