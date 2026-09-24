#!/bin/bash
HOME=/tmp PYTHONPATH=/home/liaoyuanjun/.local/lib/python3.10/site-packages ncu \
  --section-folder /usr/lib/nsight-compute/sections \
  --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
  --launch-skip 3 --launch-count 1 \
  --set full -c csv \
  -o /home/liaoyuanjun/3dgs-renderer-benchmark/results/phase-c19/ncu_tile8 \
  python3 /home/liaoyuanjun/3dgs-renderer-benchmark/scripts/c19-2/01_ncu_resource_profile.py --tile-size 8 --gpu 0
