set -e
cd /mnt/storage_pool/liaoyuanjun/acg0_mx
sed -i 's/^ALLOWED = \[1, 2, 3, 4\]/ALLOWED = [1, 2, 3, 4, 6, 7]/' acg0_watcher.py
grep -n '^ALLOWED' acg0_watcher.py
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
$PY -m py_compile acg0_watcher.py && echo WATCHER_COMPILE_OK
# restart watcher (single instance)
pkill -f acg0_watcher.py 2>/dev/null || true
sleep 2
rm -f WATCH_RESULT
setsid nohup $PY acg0_watcher.py > watch.log 2>&1 < /dev/null &
sleep 4
echo "--- watcher proc ---"; pgrep -af acg0_watcher.py | grep python
# trigger master immediately on idle GPU6
echo "IDLE idx=6 uuid=GPU-9cdad9db-1d9f-3512-9133-aff7d8086485 time=$(date +%H:%M:%S)" > WATCH_RESULT
echo "--- WATCH_RESULT ---"; cat WATCH_RESULT
