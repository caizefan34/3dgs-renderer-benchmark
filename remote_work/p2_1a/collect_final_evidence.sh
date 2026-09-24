#!/bin/bash
# Final evidence capture for the P2-1A runtime gate correctness-failure record.
set -u
OUT=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_results
cd /mnt/storage_pool/liaoyuanjun/tmp_p2_1a
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
SO_T=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
SO_B=/mnt/storage_pool/liaoyuanjun/higs_c0_cache_composed/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so

{
echo "==================== P2-1A ENTRY SMOKE MATRIX (full raw output) ===================="
echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host: $(hostname)"
echo "gpu_selection: CUDA_VISIBLE_DEVICES=3 (A100-PCIE-40GB, shared with vLLM worker; correctness probe only, no timing)"
echo
for pair in "TEST render" "TEST op" "TEST producer" "C0 render" "C0 op" "C0 producer"; do
  set -- $pair
  if [ "$1" = "TEST" ]; then SO="$SO_T"; else SO="$SO_B"; fi
  echo "=== $1 / $2 ==="
  CUDA_VISIBLE_DEVICES=3 $PY p2_1a_smoke.py --so "$SO" --tag "$1" --mode "$2" 2>&1
  echo
done
} > "$OUT/entry_smoke_matrix.log" 2>&1
tail -5 "$OUT/entry_smoke_matrix.log"

echo "==================== GPU LIVE STATE ===================="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo
nvidia-smi pmon -c 1
echo
echo "==================== DRIVER / ENV ===================="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
$PY -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'python', sys.version.split()[0])"
sha256sum "$SO_T" "$SO_B" /mnt/storage_pool/liaoyuanjun/higs_p2_1a_results/gsplat_cuda_p2_1a.so /mnt/storage_pool/liaoyuanjun/higs_p2_1a_results/higs-p2-1a-native-hierarchy-forward.patch /tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so
