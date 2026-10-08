#!/bin/bash
# Surgical: verify-clean GPUs, then run 40 iters with in-run device prints,
# polling nvidia-smi every 2s to catch the context while alive.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
cd "$D"
echo "=== pre-check: all contexts ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader || true
pkill -9 -f vos_train_driver 2>/dev/null || true
pkill -9 -f publication_trainer 2>/dev/null || true
sleep 3
echo "=== after force kill ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
rm -rf /tmp/pin_test
CUDA_VISIBLE_DEVICES=6 $PY vos_train_driver.py --scene room --iterations 40 \
  --gpu 6 --vos-start 99999 --final-eval subset --outdir /tmp/pin_test \
  > /tmp/pin_test.log 2>&1 &
BGPID=$!
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sleep 5
  echo "t=$((i*5))s: $(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | tr '\n' ' ')"
done
wait $BGPID
grep -E "vos-dbg|vos-cuda" /tmp/pin_test.log
tail -1 /tmp/pin_test.log
echo "=== index 6 uuid: $(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | sed -n '7p')"
