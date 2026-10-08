#!/bin/bash
# P14 selector: ORACLE_ONLY, both scenes sequential on GPU 4
cd /root/epic05_p14/scripts
P=/root/miniforge3/envs/gsplat/bin/python
$P p14_selector.py > /root/epic05_p14/logs/selector.log 2>&1
echo "exit=$?"
tail -8 /root/epic05_p14/logs/selector.log
