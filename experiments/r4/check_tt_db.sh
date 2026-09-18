#!/bin/bash
for s in train truck drjohnson playroom; do
  echo -n "$s cameras: "
  if [ -f "$HOME/3dgs-renderer-benchmark/data/official/mipnerf360/$s/cameras.json" ]; then
    echo "YES"
  else
    echo "NO"
  fi
  echo -n "$s images: "
  ls "$HOME/3dgs-renderer-benchmark/data/datasets/mipnerf360/$s/images/" 2>/dev/null | head -1
  echo -n "$s sfm: "
  if [ -f "$HOME/3dgs-renderer-benchmark/data/datasets/mipnerf360/$s/sparse/0/points3D.bin" ]; then
    echo "YES"
  else
    echo "NO"
  fi
done
