#!/bin/bash
W=/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat
echo '=== WORK root ls ==='
ls -la "$W" | head -40
echo '=== does WORK/gsplat exist ==='
ls -d "$W/gsplat" 2>/dev/null && echo HAS_gsplat_DIR || echo NO_gsplat_DIR
echo '=== WORK/__init__.py ==='
ls -la "$W/__init__.py" 2>/dev/null || echo NO_INIT_AT_ROOT
echo '=== WORK/gsplat/__init__.py ==='
ls -la "$W/gsplat/__init__.py" 2>/dev/null || echo NO_gsplat_INIT
echo '=== current /tmp/p3h_boot ==='
ls -la /tmp/p3h_boot/ 2>/dev/null