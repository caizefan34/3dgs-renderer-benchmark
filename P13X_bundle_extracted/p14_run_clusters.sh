#!/bin/bash
cd /root/epic05_p14/scripts
/root/miniforge3/envs/gsplat/bin/python p14_clusters.py \
  > /root/epic05_p14/logs/clusters.log 2>&1
echo "exit=$?"
tail -20 /root/epic05_p14/logs/clusters.log
