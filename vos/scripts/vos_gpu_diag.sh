#!/bin/bash
# Which processes hold which GPUs?
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
echo "---"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--- drivers holding GPUs:"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
  echo "pid=$pid: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -o 'outdir [^ ]*' || echo '?')"
done
echo "--- scene log tails:"
for s in room bicycle garden; do
  echo "== $s =="; tail -3 "/root/vos_traces/logs/vos_c0_${s}_30k.log" 2>/dev/null
done
