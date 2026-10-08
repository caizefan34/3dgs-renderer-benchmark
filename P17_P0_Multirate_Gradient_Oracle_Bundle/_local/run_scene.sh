#!/usr/bin/env bash
# P17-P0 full per-scene runner. Usage: run_scene.sh <scene>
set -u
SCENE=$1
GPU=${GPU:-3}
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/epic05_p17
cd $D

echo "[$(date -Is)] === $SCENE start (GPU $GPU) ==="
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader -i $GPU > cont_${SCENE}_gpu_before.csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > cont_${SCENE}_procs_before.csv
uptime > cont_${SCENE}_uptime_before.txt

CUDA_VISIBLE_DEVICES=$GPU $PY p17_p0_oracle.py --scene $SCENE --gpu $GPU --out $D 2>&1 | tee -a ${SCENE}_main.log
RC=$?
echo "[$(date -Is)] $SCENE main rc=$RC"

# camera set was frozen by the main script BEFORE any gradient probe.
# Run sensitivity taus (tau=1, tau=4) now; these never influence the
# primary verdict (prereg sections 1, 13-15).
CUDA_VISIBLE_DEVICES=$GPU $PY p17_p0_tau.py --scene $SCENE --tau 1.0 --out $D 2>&1 | tee -a ${SCENE}_tau1.log
echo "[$(date -Is)] $SCENE tau1 rc=$?"
CUDA_VISIBLE_DEVICES=$GPU $PY p17_p0_tau.py --scene $SCENE --tau 4.0 --out $D 2>&1 | tee -a ${SCENE}_tau4.log
echo "[$(date -Is)] $SCENE tau4 rc=$?"

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader -i $GPU > cont_${SCENE}_gpu_after.csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > cont_${SCENE}_procs_after.csv
uptime > cont_${SCENE}_uptime_after.txt
echo "[$(date -Is)] === $SCENE done ==="
