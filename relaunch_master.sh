#!/bin/bash
cd /mnt/storage_pool/liaoyuanjun/acg0_mx
pkill -f acg0_master.py 2>/dev/null
sleep 2
rm -f MASTER_RESULT.json acg0_r0/VERDICT.json ACG0_RESULTS_MX.zip
rm -rf acg0_verify acg0_verify_diag
printf 'IDLE idx=6 uuid=GPU-9cdad9db-1d9f-3512-9133-aff7d8086485 time=%s\n' "$(date +%H:%M:%S)" > WATCH_RESULT
setsid nohup /home/liaoyuanjun/miniforge3/envs/nerficg/bin/python acg0_master.py > master.log 2>&1 < /dev/null &
echo "master_launched pid=$!"
sleep 8
echo "--- master proc ---"; pgrep -af acg0_master | grep python
echo "--- WATCH_RESULT ---"; cat WATCH_RESULT
echo "--- master.log ---"; cat master.log
