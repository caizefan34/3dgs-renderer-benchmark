#!/bin/bash
# P11 environment probe on EPIC-05
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
echo "=== nsys search ==="
ls /opt/nvidia/nsight-systems*/bin/nsys 2>/dev/null || true
ls /usr/local/bin/nsys /usr/bin/nsys 2>/dev/null || true
find / -maxdepth 4 -name 'nsys' -type f 2>/dev/null | head -5
echo "=== internet check ==="
timeout 15 curl -sI https://github.com 2>&1 | head -3 || echo "CURL github FAIL"
timeout 15 curl -sI https://pypi.org 2>&1 | head -3 || echo "CURL pypi FAIL"
echo "=== gmain tree git ==="
ls -a /root/epic05_p8/gsplat-main-512d366b/ | head -10
cat /root/epic05_p8/gsplat-main-512d366b/.git-sha 2>/dev/null || true
cat /root/epic05_p8/gmain_build_identity.json 2>/dev/null | head -40
echo "=== gsplat version in tree ==="
grep -m2 'version' /root/epic05_p8/gsplat-main-512d366b/gsplat/__init__.py 2>/dev/null || true
echo "=== gaussian_model add_densification_stats ==="
sed -n '240,300p' /root/3dgs-renderer-benchmark/baseline/reference_v1/gaussian_model.py
echo "=== config key fields ==="
grep -nE 'densify|lambda_dssim|sh_degree|percent_dense|opacity_reset' /root/3dgs-renderer-benchmark/baseline/reference_v1/config.py | head -20
echo "=== datasets ==="
ls /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/ 2>/dev/null
ls /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/sparse/0/ 2>/dev/null
echo "=== gsplat rasterization signature/meta keys ==="
grep -n 'def rasterization' /root/epic05_p8/gsplat-main-512d366b/gsplat/rendering.py | head -3
grep -n 'meta\[' /root/epic05_p8/gsplat-main-512d366b/gsplat/rendering.py | head -20
