#!/bin/bash
# Gate + Stage 1/2 runner: clean-GPU gate then run_stage12.sh <scene>, detached.
# Usage: run_stage12_gated.sh <room|bicycle>
set -u
ROOT=/root/epic05_d0
SCENE="${1:?usage: run_stage12_gated.sh <room|bicycle>}"
LOG=$ROOT/logs/stage12_${SCENE}_batch.log
nohup bash -c "echo '== stage12 ${SCENE} gated launch '\$(date -Is); bash $ROOT/scripts/gpu_gate.sh 2 10 30 && bash $ROOT/scripts/run_stage12.sh $SCENE; echo '== stage12 ${SCENE} gated wrapper exit rc='\$?" >> "$LOG" 2>&1 &
echo "launched pid=$! log=$LOG"
