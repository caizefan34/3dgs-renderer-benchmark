#!/bin/bash
# Launch all 4 S0 cells in parallel, one per GPU, each running offsets 0,2,4,6.
cd /root/epic05_s0
export LD_LIBRARY_PATH=/root/miniforge3/envs/gsplat/lib:/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p cells out
pkill -f s0_cell.py 2>/dev/null
sleep 2
launch() {
  local sc=$1 ck=$2 gpu=$3
  setsid bash run_cell.sh $sc $ck 0,2,4,6 $gpu </dev/null >/tmp/full_${sc}_${ck}.out 2>&1 &
  echo "launched $sc@$ck on GPU$gpu"
}
launch room    10000 0
launch room    15000 1
launch bicycle 10000 2
launch bicycle 15000 3
sleep 3
echo "--- running ---"
ps -eo pid,cmd | grep s0_cell.py | grep -v grep | awk '{print $1, $5, $6, $7, $8}'