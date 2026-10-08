#!/bin/bash
ls /root/epic05_p17/ 2>/dev/null
echo ===
grep -n "mark_dirty\|_HIGS_DYNAMIC_SCENE\|\.reset()\|ensure_renderer" /root/epic05_p17/*.py 2>/dev/null | head -30
echo ===HANDLE_MARKDIRTY===
grep -n "def mark_dirty" -A 20 /root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py | head -60
