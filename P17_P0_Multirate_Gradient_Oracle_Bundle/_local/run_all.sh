#!/usr/bin/env bash
# P17-P0 final campaign runner.
set -u
D=/root/epic05_p17
cd $D
export GPU=3
PY=/root/miniforge3/envs/gsplat/bin/python

echo "[$(date -Is)] P17-P0 campaign start"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader > cont_campaign_gpu_start.csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > cont_campaign_procs_start.csv
uptime > cont_campaign_uptime_start.txt

echo "[$(date -Is)] === room start (GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 $PY p17_p0_oracle.py --scene room --gpu 3 --out $D 2>&1 | tee -a room_main.log
echo "[$(date -Is)] room main rc=${PIPESTATUS[0]}"
CUDA_VISIBLE_DEVICES=3 $PY p17_p0_tau.py --scene room --tau 1.0 --out $D 2>&1 | tee -a room_tau1.log
echo "[$(date -Is)] room tau1 rc=${PIPESTATUS[0]}"
CUDA_VISIBLE_DEVICES=3 $PY p17_p0_tau.py --scene room --tau 4.0 --out $D 2>&1 | tee -a room_tau4.log
echo "[$(date -Is)] room tau4 rc=${PIPESTATUS[0]}"
echo "[$(date -Is)] === room done ==="

echo "[$(date -Is)] === bicycle start (GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 $PY p17_p0_oracle.py --scene bicycle --gpu 3 --out $D 2>&1 | tee -a bicycle_main.log
echo "[$(date -Is)] bicycle main rc=${PIPESTATUS[0]}"
CUDA_VISIBLE_DEVICES=3 $PY p17_p0_tau.py --scene bicycle --tau 1.0 --out $D 2>&1 | tee -a bicycle_tau1.log
echo "[$(date -Is)] bicycle tau1 rc=${PIPESTATUS[0]}"
CUDA_VISIBLE_DEVICES=3 $PY p17_p0_tau.py --scene bicycle --tau 4.0 --out $D 2>&1 | tee -a bicycle_tau4.log
echo "[$(date -Is)] bicycle tau4 rc=${PIPESTATUS[0]}"
echo "[$(date -Is)] === bicycle done ==="

nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > cont_campaign_procs_end.csv
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader > cont_campaign_gpu_end.csv
uptime > cont_campaign_uptime_end.txt
echo "[$(date -Is)] P17-P0 campaign done"
