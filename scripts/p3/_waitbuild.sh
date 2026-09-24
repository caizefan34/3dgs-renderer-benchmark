#!/bin/bash
# Tail the current build log, printing until a fresh .so appears for V0,
# or until no build process remains.
cd /mnt/storage_pool/liaoyuanjun
so=/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/V0/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
for i in $(seq 1 55); do
  if [ -f "$so" ]; then echo "BUILD_DONE"; ls -la "$so"; exit 0; fi
  sleep 10
done
echo "TIMEOUT_550s"