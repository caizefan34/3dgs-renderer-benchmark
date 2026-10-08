#!/bin/bash
# P16-RB: frozen-recipe P12 C0 triad rerun.
# Rule basis: 500-step replay median iteration time fell outside +-5% of the
# archived P12 median (regime-mismatch statistic); prereg remedy = rerun triad.
# Sequential on one physical A100 (GPU 2); exact archived recipe, seed 42.
set -u
D=/root/epic05_p16rb
GPU=2
export CUDA_VISIBLE_DEVICES=$GPU

for SCENE in room bicycle garden; do
  OUT=$D/runs/P12_C0/$SCENE
  L=$D/logs/P12_C0_${SCENE}
  mkdir -p "$OUT"
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
    --format=csv,noheader -i $GPU > ${L}.gpu_before.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_before.csv
  uptime > ${L}.uptime_before.txt
  T0=$(date -Is)
  /root/epic05_py_wrapper.sh /root/epic05_p12/publication_trainer_p12.py --arm c0 --scene $SCENE \
    --iterations 30000 --outdir "$OUT" --gpu $GPU --lpips --final-eval all --checkpoints \
    --timing-grade PUBLICATION --eps2d 0.3 --loss fused > ${L}.stdout.log 2>&1
  RC=$?
  T1=$(date -Is)
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
    --format=csv,noheader -i $GPU > ${L}.gpu_after.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_after.csv
  uptime > ${L}.uptime_after.txt
  echo "$SCENE rc=$RC T0=$T0 T1=$T1"
done
echo "P12_C0_TRIAD_DONE $(date -Is)"
