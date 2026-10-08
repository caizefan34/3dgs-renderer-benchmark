echo "=== g3 procs ==="
ps -eo pid,etime,cmd | grep g3_gpu_micro | grep -v grep | awk "{print \$1,\$2}"
echo "=== which gpu (CUDA_VISIBLE_DEVICES in /proc) ==="
for p in $(pgrep -f g3_gpu_micro); do echo "pid $p:"; tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep CUDA_VISIBLE; done
