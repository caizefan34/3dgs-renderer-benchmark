#!/bin/bash
set +e
echo "T=$(date -u +%H:%M:%SZ)"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=30000{c++} END{print "gpus>=30GiB:", c+0}'
find /mnt/storage_pool/liaoyuanjun -maxdepth 4 -iname "*C51*manifest*" -o -maxdepth 4 -iname "*C51*candidate*" 2>/dev/null | head -3
echo DONE
