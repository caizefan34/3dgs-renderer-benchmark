#!/bin/bash
echo "=== disk ==="; df -h /root | tail -1
echo "=== gpu ==="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== selective adam ==="
grep -rn "SelectiveAdam\|selective_adam" /root/3dgs-renderer-benchmark/src /root/3dgs-renderer-benchmark/baseline /root/3dgs-renderer-benchmark/scripts 2>/dev/null | head -5
grep -rn "SelectiveAdam\|selective" /root/epic05_frozen/higs_c0_final30k_worktree --include="*.py" 2>/dev/null | head -5
echo "=== scheduler ==="
grep -n "_make_expon_lr_func\|def update_learning_rate" -A 12 /root/3dgs-renderer-benchmark/baseline/reference_v1/gaussian_model.py | head -40
echo "=== p13x ckpts ==="
ls -la /root/epic05_p13x/runs/room_diag/ckpt_16000.pt /root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt
