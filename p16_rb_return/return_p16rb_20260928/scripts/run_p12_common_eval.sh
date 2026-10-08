#!/bin/bash
# P16-RB: common-split (hold-8) render + P12-metric evaluation of the frozen
# P12 C0 triad checkpoints. Render-only; no training. Runs after the triad.
set -u
D=/root/epic05_p16rb
GPU=2
export CUDA_VISIBLE_DEVICES=$GPU

for SCENE in room bicycle garden; do
  OUT=$D/runs/P12_C0/$SCENE/common_eval
  CKPT=$D/runs/P12_C0/$SCENE/ckpt_30000.pt
  L=$D/logs/P12_C0_common_${SCENE}
  if [ ! -f "$CKPT" ]; then echo "$SCENE SKIP no ckpt"; continue; fi
  mkdir -p "$OUT"
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
    --format=csv,noheader -i $GPU > ${L}.gpu_before.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_before.csv
  T0=$(date -Is)
  EXP=$D/artifacts/expected_names_${SCENE}.json
  /root/miniforge3/envs/p16_fastergs/bin/python $D/scripts/p16_expected_names.py $SCENE "$EXP" \
    >> ${L}.stdout.log 2>&1
  /root/miniforge3/envs/gsplat/bin/python $D/scripts/p16_p12_render_common.py \
    --scene $SCENE --ckpt "$CKPT" --outdir "$OUT" --gpu $GPU \
    --expected-names-file "$EXP" >> ${L}.stdout.log 2>&1
  RC=$?
  T1=$(date -Is)
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
    --format=csv,noheader -i $GPU > ${L}.gpu_after.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_after.csv
  echo "$SCENE rc=$RC T0=$T0 T1=$T1"
done
echo "P12_C0_COMMON_EVAL_DONE $(date -Is)"
