#!/bin/bash
# P14 aggregation steps (CPU only). Usage: bash p14_run_aggs.sh <o1|o2|all>
cd /root/epic05_p14/scripts
P=/root/miniforge3/envs/gsplat/bin/python
WHAT="${1:-all}"
if [ "$WHAT" = "o1" ] || [ "$WHAT" = "all" ]; then
  $P p14_o1_agg.py 2>&1 | tail -25
fi
if [ "$WHAT" = "o2" ] || [ "$WHAT" = "all" ]; then
  $P p14_o2_agg.py 2>&1 | tail -25
fi
