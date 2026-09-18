#!/bin/bash
for s in bicycle bonsai counter flowers garden kitchen room stump treehill; do
  echo -n "mipnerf360/$s: "
  if [ -d ~/3dgs-renderer-benchmark/data/datasets/mipnerf360/$s/images ]; then
    echo -n "images=$(ls ~/3dgs-renderer-benchmark/data/datasets/mipnerf360/$s/images | wc -l) "
  else
    echo -n "NO_IMAGES "
  fi
  if [ -f ~/3dgs-renderer-benchmark/data/official/mipnerf360/$s/cameras.json ]; then
    echo -n "cameras=YES"
  else
    echo -n "cameras=NO"
  fi
  echo
done
echo "SEP"
for s in train truck drjohnson playroom; do
  echo -n "other/$s: "
  if [ -d ~/3dgs-renderer-benchmark/data/datasets/mipnerf360/$s/images ]; then
    echo -n "images=$(ls ~/3dgs-renderer-benchmark/data/datasets/mipnerf360/$s/images 2>/dev/null | wc -l) "
  else
    echo -n "NO_IMAGES "
  fi
  if [ -f ~/3dgs-renderer-benchmark/data/official/mipnerf360/$s/cameras.json ]; then
    echo -n "cameras=YES"
  else
    echo -n "cameras=NO"
  fi
  echo
done
