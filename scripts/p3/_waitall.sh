#!/bin/bash
# Wait for all 5 variant .so files. Usage: _waitall.sh [max_iters]
N="${1:-120}"
so() { echo "/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/$1/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"; }
for i in $(seq 1 "$N"); do
  all=1
  for v in V0 V1 V2 V3 V4; do
    [ -f "$(so $v)" ] || all=0
  done
  if [ "$all" -eq 1 ]; then
    echo "ALL_BUILD_DONE iter=$i"
    for v in V0 V1 V2 V3 V4; do ls -la "$(so $v)"; done
    exit 0
  fi
  if ! pgrep -f 'p3_h_accum_ceiling.py build' >/dev/null; then
    echo "BUILD_PROCESS_EXITED_BEFORE_ALL_DONE iter=$i"
    for v in V0 V1 V2 V3 V4; do [ -f "$(so $v)" ] && ls -la "$(so $v)" || echo "MISSING $v"; done
    exit 1
  fi
  sleep 15
done
echo "TIMEOUT_ALL ${N}iters"