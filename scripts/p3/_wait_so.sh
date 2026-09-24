#!/bin/bash
# Wait for a variant's .so to appear (or the build process to finish),
# then print the build log tail. Usage: _wait_so.sh <Vn> [max_iters]
V="$1"
N="${2:-36}"
so=/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/${V}/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
done=0
for i in $(seq 1 "$N"); do
  if [ -f "$so" ]; then
    echo "BUILD_DONE ${V}"
    ls -la "$so"
    done=1
    break
  fi
  # also give up if the python build process is gone but no .so
  if ! pgrep -f "p3_h_accum_ceiling.py build" >/dev/null && ! pgrep -f "build.py" >/dev/null; then
    echo "PROCESS_GONE_NO_SO ${V} (iter $i)"
    break
  fi
  sleep 10
done
[ "$done" -eq 0 ] && echo "TIMEOUT ${V}"
echo "---LOG-TAIL---"
tail -8 /tmp/p3h_build.log 2>/dev/null || true