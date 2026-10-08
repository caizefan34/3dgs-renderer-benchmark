#!/bin/bash
export LD_LIBRARY_PATH=/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
cd /root/3dgs-renderer-benchmark
/root/miniforge3/envs/gsplat/bin/python /tmp/inspect_c0_render2.py > /tmp/inspect_out.txt 2>&1
echo "EXIT=$?"
wc -l /tmp/inspect_out.txt
head -80 /tmp/inspect_out.txt
