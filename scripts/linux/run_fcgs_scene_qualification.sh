#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 12 ]; then
  echo "usage: $0 FCGS_ROOT BENCHMARK_ROOT FCGS_PYTHON BENCHMARK_PYTHON GPU LAMBDA PLY CAMERAS GT BIT_ROOT DECODED_PLY OUTPUT_DIR" >&2
  exit 2
fi

fcgs_root="$1"
benchmark_root="$2"
fcgs_python="$3"
benchmark_python="$4"
gpu="$5"
lmd="$6"
ply="$7"
cameras="$8"
ground_truth="$9"
bit_root="${10}"
decoded_ply="${11}"
output_dir="${12}"
fcgs_pythonpath="$fcgs_root/submodules/simple-knn:$fcgs_root/submodules/freqencoder:$fcgs_root/submodules/gridencoder:$fcgs_root/submodules/gridcreater:$fcgs_root/submodules/arithmetic"
torch_lib="/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/lib"

cd "$fcgs_root"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$fcgs_pythonpath" LD_LIBRARY_PATH="$torch_lib" \
  "$fcgs_python" encode_single_scene.py \
  --lmd "$lmd" --ply_path_from "$ply" --bit_path_to "$bit_root"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$fcgs_pythonpath" LD_LIBRARY_PATH="$torch_lib" \
  "$fcgs_python" decode_single_scene.py \
  --lmd "$lmd" --bit_path_from "$bit_root" --ply_path_to "$decoded_ply"

cd "$benchmark_root"
CUDA_VISIBLE_DEVICES="$gpu" "$benchmark_python" src/scripts/run_local_renderer_suite.py \
  --scene "$decoded_ply" \
  --cameras "$cameras" \
  --ground-truth-dir "$ground_truth" \
  --renderers gsplat \
  --output-dir "$output_dir" \
  --frames 100 \
  --warmup 30 \
  --repeats 5 \
  --width 1920 \
  --height 1080
