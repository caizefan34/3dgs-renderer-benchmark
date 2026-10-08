#!/bin/bash
set -e
cd /root/epic05_n1d
cp -f n1d_analyze.py N1D_PREREG.md artifacts/
/root/miniforge3/envs/gsplat/bin/python n1d_finalize.py
echo "=== artifacts ==="
ls -la artifacts/
