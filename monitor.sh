#!/bin/bash
for i in $(seq 1 1200); do
  if [ -f /mnt/storage_pool/liaoyuanjun/acg0_mx/MASTER_RESULT.json ]; then
    echo "MASTER_DONE"
    cat /mnt/storage_pool/liaoyuanjun/acg0_mx/MASTER_RESULT.json
    break
  fi
  tail -n1 /mnt/storage_pool/liaoyuanjun/acg0_mx/watch.log 2>/dev/null
  sleep 60
done
