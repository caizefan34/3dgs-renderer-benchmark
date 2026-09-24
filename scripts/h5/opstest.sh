#!/bin/bash
for E in strong_native vomp; do
  echo "== $E =="
  ~/miniforge3/envs/$E/bin/python /tmp/h5/ops.py 2>&1 | grep -E "cuda|OK|FAIL|Error" | tail -4
done