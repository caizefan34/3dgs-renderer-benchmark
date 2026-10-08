#!/usr/bin/env bash
# AB0 CHECKPOINT RECOVERY COMMANDS — READY TO RUN, NOT EXECUTED.
#
# Status: AB0 is at Stage -1 blocker (AB0_CHECKPOINT_BLOCKER).
# These commands re-derive the exact frozen-C0 5K/15K intermediate checkpoints
# so AB0 can proceed. They are a DEVIATION from the prereg's "exact checkpoint
# / Do not substitute" hard gate and REQUIRE research-lead authorization before
# running. Nothing here has been executed.
#
# Why this is a legitimate re-derivation (not a substitution):
#   - publication_trainer.py seeds torch/np/random with the frozen seed and uses
#     a frozen camera-sequence RNG (random.Random(seed)) -> deterministic trajectory.
#   - No resume/init-from-ckpt support; trainer always inits fresh from COLMAP ply
#     and saves ONLY the final checkpoint (line 658-664).
#   - Therefore a fresh run with --iterations 5000 --checkpoints reproduces the
#     exact 5K state of the frozen 30K trajectory (same for 15000).
#   - CAVEAT: GPU atomic scatter-add in densification may not be bit-identical
#     run-to-run; the original 5K/15K states were never saved, so there is nothing
#     to be bit-identical TO. Research lead must accept a seeded re-run as the
#     "exact" checkpoint.
#
# Frozen publication commands (from command.txt on EPIC-05):
#   room    c0_s43: --seed 43 --eps2d 0.3   (also c0_s42 seed 42, c0_s44 seed 44)
#   bicycle c0_s43: --seed 43 --eps2d 0.3   (also c0_s42 seed 42, c0_s44 seed 44)
#
# Prereg requires one frozen seed per scene. Default below uses seed 43 (c0_s43),
# matching the room/bicycle reference runs cited in SB0_REPORT.md.

set -euo pipefail
PYWRAP=/root/epic05_py_wrapper.sh
TRAINER=/root/3dgs-publication-epic05/publication_trainer.py
BASE=/root/epic05_ab0_recovery          # separate outdir; do NOT overwrite frozen publication runs
GPU=0

mkdir -p "$BASE"

# ---- room @ 5K and 15K (frozen seed 43) ----
$PYWRAP $TRAINER --arm c0 --scene room --iterations 5000  --outdir "$BASE/room_s43_5k"  --gpu $GPU --lpips --final-eval subset --checkpoints --timing-grade PUBLICATION --seed 43 --eps2d 0.3
$PYWRAP $TRAINER --arm c0 --scene room --iterations 15000 --outdir "$BASE/room_s43_15k" --gpu $GPU --lpips --final-eval subset --checkpoints --timing-grade PUBLICATION --seed 43 --eps2d 0.3

# ---- bicycle @ 5K and 15K (frozen seed 43) ----
$PYWRAP $TRAINER --arm c0 --scene bicycle --iterations 5000  --outdir "$BASE/bicycle_s43_5k"  --gpu $GPU --lpips --final-eval subset --checkpoints --timing-grade PUBLICATION --seed 43 --eps2d 0.3
$PYWRAP $TRAINER --arm c0 --scene bicycle --iterations 15000 --outdir "$BASE/bicycle_s43_15k" --gpu $GPU --lpips --final-eval subset --checkpoints --timing-grade PUBLICATION --seed 43 --eps2d 0.3

# Outputs: $BASE/{room,bicycle}_s43_{5k,15k}/ckpt_{5000,15000}.pt
# After recovery, verify ckpt arm/scene/iteration fields, then AB0 can run.
echo "Recovery complete. Verify ckpt_{5000,15000}.pt arm/scene/iteration, then run AB0."
