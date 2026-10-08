#!/bin/bash
cd /mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN
LOG=\
for i in \; do
  sleep 15
  if grep -qE 'CORR_END|TIMING_END|Traceback|Error|FileNotFound' \ 2>/dev/null; then
    echo DONE_POLL
    break
  fi
done
echo '=== TAIL ==='
tail -25 \
