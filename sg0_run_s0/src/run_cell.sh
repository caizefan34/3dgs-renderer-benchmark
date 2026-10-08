#!/bin/bash
SCENE=$1
CKPT=$2
OFFS=$3
GPU=${4:-0}
cd /root/epic05_s0
export LD_LIBRARY_PATH=/root/miniforge3/envs/gsplat/lib:/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p cells out
TAG="${SCENE}_${CKPT}_off${OFFS//,/-}"
rm -f cells/${TAG}.done cells/${TAG}.out cells/${TAG}.log
setsid /root/miniforge3/envs/gsplat/bin/python s0_cell.py \
  --scene $SCENE --ckpt $CKPT --outdir cells --offsets $OFFS \
  --log cells/${TAG}.log > cells/${TAG}.out 2>&1 < /dev/null
echo "EXIT=$?" >> cells/${TAG}.out
touch cells/${TAG}.done