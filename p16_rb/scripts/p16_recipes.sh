#!/bin/bash
# P16-RB frozen per-scene native recipes.
# graphdeco: upstream train_base.sh (FastGS) / run_train.sh (SADGS) scene arguments at the
# pinned commits, plus the prereg section-5 common-resolution override (-r 1).
# nerficg methods use shipped YAML configs (configs/p16_<method>_<scene>_1920.yaml).
D=/root/epic05_p16rb

native_env() {
  case $1 in
    FasterGS|FasterGSFused) echo p16_fastergs ;;
    FastGS) echo p16_fastgs ;;
    SADGS) echo p16_sadgs ;;
  esac
}

native_n() {
  case $1 in
    FasterGS|FasterGSFused|FastGS) echo 30000 ;;
    SADGS) echo 3000 ;;
  esac
}

graphdeco_recipe() {  # $1=method $2=scene $3=save_iterations
  local M=$1 SC=$2 SAVE=$3
  local BASE="train.py -s $D/common_data/mipnerf360/$SC -i images --eval -r 1"
  local R
  case $M in
    FastGS)
      case $SC in
        room)    R="--densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0008" ;;
        bicycle) R="--densification_interval 500 --optimizer_type default --test_iterations 30000 --grad_abs_thresh 0.0012" ;;
        garden)  R="--densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0008" ;;
      esac ;;
    SADGS)
      case $SC in
        room)    R="--scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --prune_iterations 1900" ;;
        bicycle) R="--optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --prune_iterations 1900" ;;
        garden)  R="--adam_eps_order 10 --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --prune_iterations 1900" ;;
      esac ;;
  esac
  echo "$BASE $R --save_iterations $SAVE"
}
