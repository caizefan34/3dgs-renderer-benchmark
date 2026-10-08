#!/bin/bash
cd /root/rf0a_laneA/FastGS
export PYTHONPATH=/root/rf0a_laneA/FastGS:$PYTHONPATH
DATA=/root/epic05_p16rb/common_data/mipnerf360
RUNS=/root/rf0a_laneA/runs
mkdir -p $RUNS /root/rf0a_laneA/logs /root/rf0a_laneA/final
LOG=/root/rf0a_laneA/logs
# clean prior outputs
rm -rf $RUNS/* /root/rf0a_laneA/final/RF0A_raw_*.json

CUDA_VISIBLE_DEVICES=0 nohup python /root/rf0a_laneA/rf0a_runner.py --method fastgs \
  -s $DATA/room -i images --eval -m $RUNS/room_fastgs \
  --mult 0.5 --eval_mult 0.5 --optimizer_type default --iterations 30000 \
  --densification_interval 500 \
  --highfeature_lr 0.02 --grad_abs_thresh 0.0008 \
  --out /root/rf0a_laneA/final/RF0A_raw_room_fastgs.json > $LOG/room_fastgs.log 2>&1 &
echo "room_fastgs PID $!"

CUDA_VISIBLE_DEVICES=2 nohup python /root/rf0a_laneA/rf0a_runner.py --method fastgs \
  -s $DATA/garden -i images --eval -m $RUNS/garden_fastgs \
  --mult 0.5 --eval_mult 0.5 --optimizer_type default --iterations 30000 \
  --densification_interval 500 \
  --highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0008 \
  --out /root/rf0a_laneA/final/RF0A_raw_garden_fastgs.json > $LOG/garden_fastgs.log 2>&1 &
echo "garden_fastgs PID $!"

CUDA_VISIBLE_DEVICES=4 nohup python /root/rf0a_laneA/rf0a_runner.py --method fastgs \
  -s $DATA/bicycle -i images --eval -m $RUNS/bicycle_fastgs \
  --mult 0.5 --eval_mult 0.5 --optimizer_type default --iterations 30000 \
  --densification_interval 500 \
  --grad_abs_thresh 0.0008 \
  --out /root/rf0a_laneA/final/RF0A_raw_bicycle_fastgs.json > $LOG/bicycle_fastgs.log 2>&1 &
echo "bicycle_fastgs PID $!"

CUDA_VISIBLE_DEVICES=1 nohup python /root/rf0a_laneA/rf0a_runner.py --method vanilla \
  -s $DATA/room -i images --eval -m $RUNS/room_vanilla \
  --mult 1.0 --eval_mult 1.0 --iterations 30000 \
  --densification_interval 100 --densify_grad_threshold 0.0002 --percent_dense 0.01 \
  --densify_from_iter 500 --densify_until_iter 15000 --opacity_reset_interval 3000 \
  --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 30000 \
  --position_lr_delay_mult 0.01 --feature_lr 0.0025 --highfeature_lr 0.0025 \
  --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --lambda_dssim 0.2 \
  --out /root/rf0a_laneA/final/RF0A_raw_room_vanilla.json > $LOG/room_vanilla.log 2>&1 &
echo "room_vanilla PID $!"

CUDA_VISIBLE_DEVICES=3 nohup python /root/rf0a_laneA/rf0a_runner.py --method vanilla \
  -s $DATA/garden -i images --eval -m $RUNS/garden_vanilla \
  --mult 1.0 --eval_mult 1.0 --iterations 30000 \
  --densification_interval 100 --densify_grad_threshold 0.0002 --percent_dense 0.01 \
  --densify_from_iter 500 --densify_until_iter 15000 --opacity_reset_interval 3000 \
  --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 30000 \
  --position_lr_delay_mult 0.01 --feature_lr 0.0025 --highfeature_lr 0.0025 \
  --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --lambda_dssim 0.2 \
  --out /root/rf0a_laneA/final/RF0A_raw_garden_vanilla.json > $LOG/garden_vanilla.log 2>&1 &
echo "garden_vanilla PID $!"

CUDA_VISIBLE_DEVICES=5 nohup python /root/rf0a_laneA/rf0a_runner.py --method vanilla \
  -s $DATA/bicycle -i images --eval -m $RUNS/bicycle_vanilla \
  --mult 1.0 --eval_mult 1.0 --iterations 30000 \
  --densification_interval 100 --densify_grad_threshold 0.0002 --percent_dense 0.01 \
  --densify_from_iter 500 --densify_until_iter 15000 --opacity_reset_interval 3000 \
  --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 30000 \
  --position_lr_delay_mult 0.01 --feature_lr 0.0025 --highfeature_lr 0.0025 \
  --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --lambda_dssim 0.2 \
  --out /root/rf0a_laneA/final/RF0A_raw_bicycle_vanilla.json > $LOG/bicycle_vanilla.log 2>&1 &
echo "bicycle_vanilla PID $!"

echo "ALL_LAUNCHED"
