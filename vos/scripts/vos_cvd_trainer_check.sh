#!/bin/bash
# What does the frozen trainer do with --gpu / CUDA_VISIBLE_DEVICES?
echo "=== trainer CVD handling ==="
grep -n "CUDA_VISIBLE\|set_device\|args.gpu\|\.cuda(" /root/epic05_p13x/publication_trainer_p13x.py | head -20
echo "=== reference_v1 config/main CVD ==="
grep -rn "CUDA_VISIBLE\|set_device" /root/3dgs-renderer-benchmark/baseline/reference_v1/*.py 2>/dev/null | head -10
echo "=== driver module-level code (before main) ==="
awk '/^def main/,0{exit} {print NR": "$0}' /root/vos_traces/vos_train_driver.py | grep -v "^.*#" | grep -vE "^\s*$" | tail -25
