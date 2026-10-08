#!/bin/bash
ulimit -s 65536 2>/dev/null
echo "STACK=$(ulimit -s)"
export TMPDIR=/dev/shm TEMP=/dev/shm TMP=/dev/shm HOME=/dev/shm/ncu_home
timeout 400 /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10 /mnt/storage_pool/liaoyuanjun/fused_profile0/ATOM0/scripts/debug_atom0.py > /dev/shm/dbg12.log 2>&1
echo "EXIT_RC=$?" >> /dev/shm/dbg12.log
