#!/bin/bash
for E in strong_native anysplat chorus vomp nerficg; do
  PY=~/miniforge3/envs/$E/bin/python
  echo "== $E =="
  $PY /tmp/h5/abi.py 2>&1 | grep -v torch. | tail -6
done