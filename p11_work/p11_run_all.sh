#!/bin/bash
# P11 runner: profiler-only opportunity gate on EPIC-05 (clean A100-SXM4 policy).
#
# GPU policy: GPU 0 for all timed gates (refuses if not clean). State generation
# (NOT a 30K campaign; 14K faithful replay) runs in parallel on GPUs 0/1/2.
# Gates 4-7 depend on dumps from earlier gates; stage order is fixed.
set -euo pipefail
export PY=/root/miniforge3/envs/gsplat/bin/python
mkdir -p /root/epic05_p11/dumps /root/epic05_p11/artifacts /root/epic05_p11/logs
cd /root/epic05_p11

STAGE=${1:-all}

run_stage() { echo "=== P11 STAGE: $1 $(date -u +%H:%M:%S) ==="; }

if [ "$STAGE" = "all" ] || [ "$STAGE" = "gate0" ]; then
  run_stage "gate0 identity"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gate0_identity.py --gpu 0 \
    --out dumps/P11_SOURCE_IDENTITY.json 2>&1 | tee logs/gate0.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stategen" ]; then
  run_stage "stategen (pre-15K states, 14K faithful replay; NOT a 30K campaign)"
  for s in room bicycle garden; do
    mkdir -p stategen/$s
  done
  CUDA_VISIBLE_DEVICES=0 $PY p11_stategen.py --scene room    --gpu 0 \
    --outdir stategen/room    > logs/stategen_room.log 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 $PY p11_stategen.py --scene bicycle --gpu 1 \
    --outdir stategen/bicycle > logs/stategen_bicycle.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=2 $PY p11_stategen.py --scene garden  --gpu 2 \
    --outdir stategen/garden  > logs/stategen_garden.log 2>&1 &
  P2=$!
  wait $P0 $P1 $P2
  run_stage "stategen done"
  tail -n 3 logs/stategen_room.log logs/stategen_bicycle.log logs/stategen_garden.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "L1" ]; then
  run_stage "gate L1 equivalence"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gateL1_equivalence.py --gpu 0 \
    --out dumps/P11_LOSS_EQUIVALENCE.json 2>&1 | tee logs/L1.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "L2" ]; then
  run_stage "gate L2 loss-only timing"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gateL2_loss_timing.py --gpu 0 \
    --out dumps/P11_LOSS_TIMING.json 2>&1 | tee logs/L2.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "L3" ]; then
  run_stage "gate L3 renderer backward w/ external adjoint"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gateL3_renderer_bwd.py --gpu 0 \
    --out dumps/P11_L3_RENDERER_BWD.json 2>&1 | tee logs/L3.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "closure" ]; then
  run_stage "L3 companion: additive closure (streaming protocol)"
  CUDA_VISIBLE_DEVICES=0 $PY p11_closure_streaming.py --gpu 0 \
    --out dumps/P11_L3_CLOSURE_STREAMING.json 2>&1 | tee logs/closure.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "L4" ]; then
  run_stage "gate L4 kernel profile"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gateL4_profile.py --gpu 0 \
    --out dumps/P11_LOSS_KERNEL_PROFILE.json 2>&1 | tee logs/L4.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "D1" ]; then
  run_stage "gate D1 densification stats"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gateD1_densification.py --gpu 0 \
    --out dumps/P11_DENSIFICATION_TIMING.json 2>&1 | tee logs/D1.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "T" ]; then
  run_stage "gate T active interactions"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gateT_interactions.py --gpu 0 \
    --out dumps/P11_ACTIVE_INTERACTION_AUDIT.json 2>&1 | tee logs/T.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "O" ]; then
  run_stage "gate O optimizer"
  CUDA_VISIBLE_DEVICES=0 $PY p11_gateO_optimizer.py --gpu 0 \
    --out dumps/P11_OPTIMIZER_TIMING.json 2>&1 | tee logs/O.log
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "assemble" ]; then
  run_stage "assemble candidate matrix + verdict + report"
  $PY p11_assemble.py 2>&1 | tee logs/assemble.log
fi

run_stage "DONE ($STAGE)"
