#!/bin/bash
cd /root/epic05_dpc0/harness
PY=/root/miniforge3/envs/gsplat/bin/python
for cell in room:10000 room:15000 bicycle:10000 bicycle:15000; do
  scene=${cell%:*}
  it=${cell#*:}
  for gpu in 1 2 3; do
    line=$(timeout 300 $PY -u dpc0_cross.py $gpu $scene $it 2>>cross_err.log | grep "^{")
    echo "$cell gpu=$gpu -> $line"
    echo "$cell gpu=$gpu -> $line" >> cross_hashes.log
  done
done
echo "CROSS_DONE"
