#!/bin/bash
# ET0 cell pipeline: sweep already done; compute cohort tau_H, safestep, aggregate.
set -e
cd /root/epic05_moonshot/et0
P=/root/miniforge3/envs/gsplat/bin/python
OUT=$1
CELL=$2
TAU_H=$3
echo "=== tau_H for $CELL = $TAU_H ==="
echo "=== SAFESTEP ==="
$P et0_cell.py --scene $(echo $CELL | cut -d_ -f1) --ckpt $(echo $CELL | cut -d_ -f2) --outdir $OUT --mode safestep --tau-h $TAU_H 2>&1 | grep -E 'cam [0-9]|safestep done|Error|Traceback' | tail -12
echo "=== AGGREGATE ==="
$P et0_cell.py --outdir $OUT --mode aggregate --cells $CELL 2>&1 | tail -40
