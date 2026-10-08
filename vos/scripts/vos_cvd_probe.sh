#!/bin/bash
# Root-cause the CVD override: inspect running driver processes' environment
# and the driver file on the host.
echo "=== driver file has the patch? ==="
grep -n "CUDA_VISIBLE_DEVICES" /root/vos_traces/vos_train_driver.py
echo "=== python processes ==="
ps aux | grep -v grep | grep vos_train_driver | awk '{print $2}' | while read p; do
  echo "pid $p CVD: $(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep CUDA_VISIBLE || echo UNSET)"
done
echo "=== login shell env ==="
env | grep -i cuda || echo "no cuda in env"
echo "=== probe: does CVD=3 land on UUID of index 3? ==="
CUDA_VISIBLE_DEVICES=3 /root/miniforge3/envs/gsplat/bin/python -c "
import torch, threading, subprocess, os
t = torch.zeros(1024**3 // 4, device='cuda')  # 1GB to be visible
def probe():
    out = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout
    mypid = os.getpid()
    for line in out.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if int(parts[0]) == mypid:
            print('probe landed on UUID', parts[1], 'mem', parts[2])
threading.Thread(target=probe).start()
" 2>&1 | grep -v Warning
echo "=== uuid map ==="
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
