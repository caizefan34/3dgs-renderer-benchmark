#!/bin/bash
P=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
echo '=== installed gsplat ==='
$P -c "import gsplat; print(gsplat.__file__)"
echo '=== worktree kernels dir ==='
ls /mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/
echo '=== find _backend.py ==='
find /mnt/storage_pool/liaoyuanjun -path '*experimental/render/kernels/_backend.py' 2>/dev/null
echo '=== worktree kernels __init__ ==='
tail -30 /mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/__init__.py