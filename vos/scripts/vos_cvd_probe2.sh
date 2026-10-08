#!/bin/bash
# Decisive: does CUDA_VISIBLE_DEVICES=N actually pin to physical GPU N here?
PY=/root/miniforge3/envs/gsplat/bin/python
for G in 3 5; do
  echo "=== CVD=$G probe ==="
  CUDA_VISIBLE_DEVICES=$G $PY - <<'EOF' 2>&1 | grep -v Warning
import torch, subprocess, os
x = torch.zeros(256 * 1024 * 1024, device="cuda")  # 1 GiB, keep context alive
out = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
     "--format=csv,noheader"], capture_output=True, text=True).stdout
mine = str(os.getpid())
host_pids = os.listdir("/proc")  # container ns
for line in out.strip().split("\n"):
    parts = [p.strip() for p in line.split(",")]
    if parts[0] == mine:
        print(f"my pid {mine} -> UUID {parts[1]} mem {parts[2]}")
        break
else:
    # pid namespaces differ: print all and match by the fresh 1GiB allocation
    print("all contexts now:", out.strip())
EOF
done
nvidia-smi --query-gpu=index,uuid --format=csv,noheader | sed -n '1p;4p;6p'
