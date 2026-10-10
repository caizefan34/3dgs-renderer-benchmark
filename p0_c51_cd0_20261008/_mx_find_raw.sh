#!/bin/bash
set +e
ROOT=/mnt/storage_pool/liaoyuanjun
echo "=== C51 JOINT output dir ==="
ls -ld $ROOT/CVPR27_C51_JOINT_MX 2>/dev/null || echo "absent"
echo "=== confirmatory_accel15 raw results ==="
find $ROOT -maxdepth 6 -iname "*confirmatory*accel15*" 2>/dev/null | head -20
echo "=== per-run raw jsons (accel15) ==="
find $ROOT -maxdepth 7 -path "*accel15*" -name "*.json" 2>/dev/null | head -20
echo "=== C51 stage5 / phase-c51 result jsons ==="
find $ROOT -maxdepth 7 -iname "*phase-c51-stage5*" 2>/dev/null | head -20
find $ROOT -maxdepth 7 -iname "*c51*final*decision*" 2>/dev/null | head -10
echo "=== higs-paper-protocol ==="
find $ROOT -maxdepth 6 -iname "*higs-paper-protocol*" 2>/dev/null | head -10
find $ROOT -maxdepth 6 -iname "*higs_paper_protocol*" 2>/dev/null | head -10
echo "=== fused_prune10_rclip05 impl ==="
find $ROOT -maxdepth 7 -iname "*fused_prune10*" -o -iname "*prune10_rclip05*" 2>/dev/null | head -10
echo "=== MipNeRF360 datasets (room/bicycle/garden) ==="
for d in room bicycle garden; do echo "-- $d --"; find $ROOT -maxdepth 5 -name "$d" -type d 2>/dev/null | head -3; done
echo "=== rclone/remotes with capacity ==="
which rclone 2>/dev/null && rclone listremotes 2>/dev/null | head
echo "=== home fs (for small audit files) ==="
df -B1 $HOME 2>/dev/null | tail -1
echo "=== DONE ==="
