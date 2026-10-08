#!/bin/bash
# P14 report + verdict + manifest + contamination (CPU only)
cd /root/epic05_p14/scripts
P=/root/miniforge3/envs/gsplat/bin/python
$P p14_report.py 2>&1 | tail -5
echo "--- verdict ---"
cat /root/epic05_p14/artifacts/P14_VERDICT.json | head -40
