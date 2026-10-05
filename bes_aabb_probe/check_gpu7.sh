#!/bin/bash
echo "=== GPU7 compute apps ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep a191b22f
echo "=== pids cmdline ==="
for p in $(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader); do
  echo "PID=$p : $(cat /proc/$p/cmdline 2>/dev/null | tr '\0' ' ')"
done
