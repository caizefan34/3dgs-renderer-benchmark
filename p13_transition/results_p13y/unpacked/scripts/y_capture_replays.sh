#!/bin/bash
# P13-Y Sec.4 full capture replays: 0 -> 17000 per scene, capture at the frozen
# pair-plan endpoint union only. GPU 2 (clean physical GPU; see contamination log).
set -u
cd /root/epic05_p13y
PY=/root/miniforge3/envs/gsplat/bin/python
TR=publication_trainer_p13y_capture.py
COMMON="--arm c0 --iterations 17000 --gpu 2 --lpips --final-eval all --timing-grade PUBLICATION --eps2d 0.3 --loss fused"

for SCENE in room bicycle garden; do
  ITERS=$(python3 -c "import json;print(\",\".join(str(i) for i in json.load(open(\"artifacts/P13Y_PAIR_PLAN_${SCENE}.json\"))[\"capture_iterations_union\"]))")
  R=runs/y_capture_${SCENE}
  echo "== $SCENE capture_iters=$ITERS =="
  bash p13y_unit.sh y_capture_${SCENE} logs/y_capture_${SCENE}.log -- $PY $TR $COMMON \
    --scene $SCENE --outdir $R \
    --p13y-cam-log $R/cam_log.json --p13y-iter-log $R/p13y_iter_log.json \
    --p13y-capture-dir $R/states --p13y-parity-dir $R/probes \
    --p13y-capture-iters "$ITERS" --p13y-parity-iters "$ITERS" \
    --p13y-source-identity $R/p13y_capture_identity.json
  echo "${SCENE}_RC=$?"
done
echo ALL_REPLAYS_DONE
