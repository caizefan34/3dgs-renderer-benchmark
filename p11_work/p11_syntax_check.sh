#!/bin/bash
cd /root/epic05_p11
for f in *.py; do
  if /root/miniforge3/envs/gsplat/bin/python -m py_compile "$f"; then
    echo "OK $f"
  else
    echo "FAIL $f"
  fi
done
