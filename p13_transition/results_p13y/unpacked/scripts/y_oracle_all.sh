#!/bin/bash
# P13-Y Sec.5-8 runner: per scene, the adapter parity gate over that scene's
# 17K capture probes, then Y1/Y2/Y3 over all 8 frozen pairs. STOP on parity fail.
set -u
cd /root/epic05_p13y
PY=/root/miniforge3/envs/gsplat/bin/python

for SCENE in room bicycle garden; do
  R=runs/y_capture_${SCENE}
  bash p13y_unit.sh y_parity_${SCENE} logs/y_parity_${SCENE}.log -- $PY p13y_oracle.py \
    --mode parity --scene $SCENE --gpu 2 --run-dir $R/probes \
    --out artifacts/P13Y_ADAPTER_PARITY_${SCENE}.json
  RC=$?
  echo "PARITY_${SCENE}_RC=$RC"
  if [ $RC -ne 0 ]; then
    echo "STOP_ADAPTER_PARITY_FAILED scene=$SCENE"
    exit 3
  fi
  bash p13y_unit.sh y_oracle_${SCENE} logs/y_oracle_${SCENE}.log -- $PY p13y_oracle.py \
    --mode oracle --scene $SCENE --gpu 2 --capture-dir $R/states \
    --plan artifacts/P13Y_PAIR_PLAN_${SCENE}.json \
    --outdir pair_results/${SCENE}
  echo "ORACLE_${SCENE}_RC=$?"
done
echo ALL_ORACLE_DONE
