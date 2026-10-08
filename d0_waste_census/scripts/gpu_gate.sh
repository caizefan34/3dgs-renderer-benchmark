#!/bin/bash
# Clean-GPU gate: usage gpu_gate.sh <physical_gpu> <n_checks> <interval_s>
set -euo pipefail
G=${1:-2}; N=${2:-10}; IV=${3:-30}
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G")
echo "gate: gpu=$G uuid=$uuid checks=$N interval=${IV}s start=$(date -Is)"
for i in $(seq 1 "$N"); do
  line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$G")
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$G" || true)
  mu=$(echo "$line" | cut -d, -f1 | tr -d ' ')
  gu=$(echo "$line" | cut -d, -f2 | tr -d ' ')
  echo "  check $i/$N mem=${mu}MiB util=${gu}% apps='${apps}'"
  if [ -n "$apps" ]; then echo "GATE FAIL: compute apps present"; exit 3; fi
  if [ "$mu" -gt 4 ] || [ "$gu" -gt 0 ]; then echo "GATE FAIL: busy"; exit 4; fi
  [ "$i" -lt "$N" ] && sleep "$IV"
done
echo "GATE PASS $(date -Is)"
