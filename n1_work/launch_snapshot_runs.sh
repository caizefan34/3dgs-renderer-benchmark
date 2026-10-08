#!/bin/bash
# N1 Stage 0: launch snapshot training runs (room GPU0, bicycle GPU1)
set -eu
mkdir -p /root/epic05_n1/runs /root/epic05_n1/scripts /root/epic05_n1/artifacts /root/epic05_n1/logs
if [ -f /root/epic05_n1_upload_tmp.py ]; then
  mv /root/epic05_n1_upload_tmp.py /root/epic05_n1/publication_trainer_p12_n1snap.py
fi
sha256sum /root/epic05_n1/publication_trainer_p12_n1snap.py
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

launch() {
  local scene=$1 gpu=$2
  local outdir=/root/epic05_n1/runs/n1snap_c0f_s42_${scene}
  mkdir -p "$outdir"
  if [ -f "$outdir/EPIC05_SUCCESS" ]; then
    echo "SKIP $scene (already done)"
    return 0
  fi
  echo "LAUNCHING $scene on physical GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup /root/epic05_py_wrapper.sh \
    /root/epic05_n1/publication_trainer_p12_n1snap.py \
    --arm c0 --scene "$scene" --iterations 25000 \
    --outdir "$outdir" --gpu 0 --final-eval subset \
    --timing-grade FUNCTIONAL_ONLY --eps2d 0.3 --loss fused \
    --snapshot-iterations 10000,15000,20000,25000 \
    < /dev/null > "$outdir/stdout.log" 2>&1 &
  echo "LAUNCHED $scene on physical GPU$gpu pid=$!"
}

# contamination snapshot before campaign
nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,utilization.gpu --format=csv,noheader > /root/epic05_n1/cont_gpu_start.csv

launch room 0
launch bicycle 1
sleep 5
echo "=== after 5s ==="
ps aux | grep n1snap | grep -v grep | awk '{print $2, $11, $12, $13, $14, $15, $16}'
tail -5 /root/epic05_n1/runs/n1snap_c0f_s42_room/stdout.log || true
tail -5 /root/epic05_n1/runs/n1snap_c0f_s42_bicycle/stdout.log || true
