#!/bin/bash
# P2-1A smoke matrix: TEST vs C0 across render/op/producer entry points.
set -u
cd /mnt/storage_pool/liaoyuanjun/tmp_p2_1a
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
SO_T=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
SO_B=/mnt/storage_pool/liaoyuanjun/higs_c0_cache_composed/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
GPU=${GPU:-3}

run() {
  local tag="$1"; local mode="$2"; local so="$3"
  echo "=== $tag / $mode ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY p2_1a_smoke.py --so "$so" --tag "$tag" --mode "$mode" 2>&1 \
    | grep -E 'SMOKE_RESULT|FAILED:|OK |renderer created|torch 2' || echo "(no output captured)"
}

run TEST op "$SO_T"
run TEST producer "$SO_T"
run C0 render "$SO_B"
run C0 op "$SO_B"
run C0 producer "$SO_B"
