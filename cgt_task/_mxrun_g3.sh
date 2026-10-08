cd /home/liaoyuanjun/cgt_realscene/final
bash launch_g3_fix.sh
sleep 3
echo "=== g3 procs ==="
ps -eo pid,etime,cmd | grep g3_gpu_micro | grep -v grep | awk "{print \$1,\$2,\$NF}"
