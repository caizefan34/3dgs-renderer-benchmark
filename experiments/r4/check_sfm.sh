#!/bin/bash
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill train truck drjohnson playroom; do
  bin="$HOME/3dgs-renderer-benchmark/data/datasets/mipnerf360/$scene/sparse/0/points3D.bin"
  ply="$HOME/3dgs-renderer-benchmark/data/datasets/mipnerf360/$scene/sparse/0/points3D.ply"
  if [ -f "$bin" ]; then
    echo "$scene: BIN $(stat -c%s "$bin") bytes"
  elif [ -f "$ply" ]; then
    echo "$scene: PLY $(stat -c%s "$ply") bytes"
  else
    # Try benchmark_data
    bp="/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/$scene/sparse/0/points3D.ply"
    bt="/mnt/storage_pool/liaoyuanjun/benchmark_data/tanksandtemples/$scene/sparse/0/points3D.ply"
    bd="/mnt/storage_pool/liaoyuanjun/benchmark_data/deepblending/$scene/sparse/0/points3D.ply"
    if [ -f "$bp" ]; then echo "$scene: BENCH-PLY $(stat -c%s "$bp") bytes"; 
    elif [ -f "$bt" ]; then echo "$scene: T&T-PLY $(stat -c%s "$bt") bytes";
    elif [ -f "$bd" ]; then echo "$scene: DB-PLY $(stat -c%s "$bd") bytes";
    else echo "$scene: MISSING"; fi
  fi
done
