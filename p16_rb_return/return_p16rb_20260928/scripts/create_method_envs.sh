#!/bin/bash
# P16-RB: create p16_fastgs and p16_sadgs by cloning the p16_fastergs env
# (python 3.11 / torch 2.8.0+cu128), then freeze pip lists.
set -u
L=/root/epic05_p16rb/logs
for ENV in p16_fastgs p16_sadgs; do
  if [ -d /root/miniforge3/envs/$ENV ]; then echo "$ENV exists, skip"; continue; fi
  echo "=== clone $ENV $(date -Is)"
  /root/miniforge3/bin/conda create -y -n $ENV --clone p16_fastergs
  echo "clone rc=$?"
done
echo ENV_CLONES_DONE $(date -Is)
