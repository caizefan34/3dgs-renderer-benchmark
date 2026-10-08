#!/bin/bash
set -e
mkdir -p /root/epic05_p14/scripts
cd /root/epic05_p14/scripts
for f in p14_common.py p14_stage0_identity.py p14_clusters.py p14_selector.py p14_o1.py p14_o1_agg.py p14_o2_agg.py p14_o3.py p14_report.py; do
  /root/miniforge3/envs/gsplat/bin/python -m py_compile "$f" && echo "OK $f"
done
