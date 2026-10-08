#!/bin/bash
# P16-RB Stage-0 preflight driver.
# Runs the four method preflights in parallel, one per physical A100 (3/4/5/6).
# Preflights are untimed; the prereg GPU-exclusivity rule governs compared runs only.
set -u
D=/root/epic05_p16rb
PY=/root/miniforge3/envs/p16_fastergs/bin/python
L=$D/logs

$PY $D/scripts/p16_stage0_nerficg.py --method FasterGS --scene room --gpu 3 --iterations 100 \
  > $L/stage0_runner_FasterGS.log 2>&1 &
P1=$!

$PY $D/scripts/p16_stage0_nerficg.py --method FasterGSFused --scene room --gpu 5 --iterations 100 \
  > $L/stage0_runner_FasterGSFused.log 2>&1 &
P2=$!

$PY $D/scripts/p16_stage0_graphdeco.py --method FastGS --repo $D/src/FastGS --env p16_fastgs \
  --scene room --gpu 4 --smoke-iterations 700 \
  --native-args "--densification_interval 500 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0008" \
  > $L/stage0_runner_FastGS.log 2>&1 &
P3=$!

$PY $D/scripts/p16_stage0_graphdeco.py --method SADGS --repo $D/src/SADGS --env p16_sadgs \
  --scene room --gpu 6 --smoke-iterations 700 \
  --native-args "--scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --prune_iterations 1900" \
  > $L/stage0_runner_SADGS.log 2>&1 &
P4=$!

echo "pids FasterGS=$P1 FasterGSFused=$P2 FastGS=$P3 SADGS=$P4"
wait $P1; echo "FasterGS rc=$?"
wait $P2; echo "FasterGSFused rc=$?"
wait $P3; echo "FastGS rc=$?"
wait $P4; echo "SADGS rc=$?"
echo "STAGE0_ALL_DONE $(date -Is)"
