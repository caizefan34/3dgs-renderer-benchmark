#!/bin/bash
DIR=/home/liaoyuanjun/3dgs-renderer-benchmark
LOG=$DIR/results/phase-c30
export PYTHONPATH=$DIR/src:$DIR:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 nohup python3 $DIR/scripts/phase-c30/c30_a_param_frequency.py --out $LOG/c30_a_param_frequency.json --steps 500 > $LOG/c30_a.log 2>&1 &
echo A GPU0

CUDA_VISIBLE_DEVICES=1 nohup python3 $DIR/scripts/phase-c30/c30_b_birth_warmstart.py --out $LOG/c30_b_birth_warmstart.json --steps 500 > $LOG/c30_b.log 2>&1 &
echo B GPU1

CUDA_VISIBLE_DEVICES=2 nohup python3 $DIR/scripts/phase-c30/c30_c_renderer_optimizer.py --out $LOG/c30_c_renderer_optimizer.json --steps 500 > $LOG/c30_c.log 2>&1 &
echo C GPU2

CUDA_VISIBLE_DEVICES=3 nohup python3 $DIR/scripts/phase-c30/c30_d_regime.py --out $LOG/c30_d_regime.json --steps 500 > $LOG/c30_d.log 2>&1 &
echo D GPU3

CUDA_VISIBLE_DEVICES=4 nohup python3 $DIR/scripts/phase-c30/c30_e_loss_frequency.py --out $LOG/c30_e_loss_frequency.json --steps 500 > $LOG/c30_e.log 2>&1 &
echo E GPU4

CUDA_VISIBLE_DEVICES=5 nohup python3 $DIR/scripts/phase-c30/c30_f_cross_iteration.py --out $LOG/c30_f_cross_iteration.json --steps 500 > $LOG/c30_f.log 2>&1 &
echo F GPU5

CUDA_VISIBLE_DEVICES=6 nohup python3 $DIR/scripts/phase-c30/c30_g_multigpu.py --out $LOG/c30_g_multigpu.json --steps 200 > $LOG/c30_g.log 2>&1 &
echo G GPU6

CUDA_VISIBLE_DEVICES=7 nohup python3 $DIR/scripts/phase-c30/c30_h_precision.py --out $LOG/c30_h_precision.json --steps 500 > $LOG/c30_h.log 2>&1 &
echo H GPU7

echo "All launched"
