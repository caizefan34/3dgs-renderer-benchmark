#!/bin/bash
set +e
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
echo "=== gaussian_model.py anywhere in repo ==="
find $REPO -name "gaussian_model.py" -not -path "*/.git/*" 2>/dev/null | head -10
echo "=== dataset.py anywhere ==="
find $REPO -name "dataset.py" -not -path "*/.git/*" 2>/dev/null | head -10
echo "=== loss.py / d_ssim anywhere ==="
find $REPO -name "loss.py" 2>/dev/null | grep -v ".git" | head -10
echo "=== what benchmark.py imports ==="
grep -nE "^from |^import " $REPO/scripts/phase-c51-stage5/benchmark.py | head -20
echo "=== does GTDataset live elsewhere? grep class defs ==="
grep -rIl "class GTDataset" $REPO --include=*.py 2>/dev/null | head -5
grep -rIl "class GaussianModel" $REPO --include=*.py 2>/dev/null | head -5
grep -rIl "def d_ssim_loss" $REPO --include=*.py 2>/dev/null | head -5
echo "=== epic05/phase7 dir (benchmark.py inserts this on sys.path) ==="
ls -d $REPO/scripts/epic05/phase7 2>/dev/null && ls $REPO/scripts/epic05/phase7 2>/dev/null | head
echo "DONE"
