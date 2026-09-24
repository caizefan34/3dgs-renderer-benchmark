#!/bin/bash
# Check PLY SH coefficient count
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=1
$PY -c "
from plyfile import PlyData
import numpy as np
ply = PlyData.read('/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply')
v = ply['vertex']
props = [p.name for p in v.properties]
dc = [p for p in props if p.startswith('f_dc')]
rest = [p for p in props if p.startswith('f_rest')]
print(f'f_dc count: {len(dc)}')
print(f'f_rest count: {len(rest)}')
total_sh = len(dc) + len(rest)
k_per_channel = total_sh // 3
print(f'Total SH coeffs: {total_sh}, K per channel: {k_per_channel}')
import math
deg = int(math.sqrt(k_per_channel)) - 1
print(f'SH degree: {deg}')
print(f'N vertices: {len(v)}')
# Show first few rest props
print(f'First 5 f_rest: {rest[:5]}')
print(f'Last 5 f_rest: {rest[-5:]}')
"
