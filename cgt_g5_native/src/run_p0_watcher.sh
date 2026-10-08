#!/usr/bin/env bash
# CGT-G5 P0 watcher: wait for a clean/exclusive MX GPU, then run the FROZEN F0 harness
# to re-confirm the ~2.99x mixed forward speed. Fail-closed idle gate.
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
ROOT=/home/liaoyuanjun/cgt_g5_native
F0=/home/liaoyuanjun/cgt_f0_fused_cuda
WORKLOAD=$F0/results/workload.npz
SO=$F0/src/f0_kernels.so
OUT=$ROOT/results
mkdir -p "$OUT"
echo "[watcher] start $(date)  budget=7200s window=60s util<=5% mem<=1500MiB no-foreign-pid"
timeout 7300 $PY "$ROOT/src/g5_p0.py" --mode measure \
  --workload "$WORKLOAD" --so "$SO" --out "$OUT" --pybin "$PY" \
  --util-max 5.0 --mem-mib-max 6144.0 --window-s 60 --poll-s 20 --budget-s 7200 \
  --repeats 300 --warmup 60
rc=$?
echo "[watcher] exit code $rc $(date)"
exit $rc
