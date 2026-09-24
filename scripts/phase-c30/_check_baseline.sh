#!/bin/bash
cd ~/3dgs-renderer-benchmark
echo "=== GIT COMMIT ==="
git rev-parse HEAD
echo "=== GIT BRANCH ==="
git rev-parse --abbrev-ref HEAD
echo "=== GSPLAT VERSION ==="
python3 -c "import gsplat; print(gsplat.__version__ if hasattr(gsplat, '__version__') else 'no __version__ attr')"
echo "=== GSPLAT PATH ==="
python3 -c "import gsplat; print(gsplat.__file__)"
echo "=== TORCH VERSION ==="
python3 -c "import torch; print(torch.__version__)"
echo "=== CUDA VERSION ==="
python3 -c "import torch; print(torch.version.cuda)"
echo "=== GPU COUNT ==="
python3 -c "import torch; print(torch.cuda.device_count())"
echo "=== GPU NAMES ==="
python3 -c "import torch; [print(torch.cuda.get_device_properties(i).name) for i in range(torch.cuda.device_count())]"
echo "=== PYTHON ==="
which python3
echo "=== LS src/ ==="
ls src/
echo "=== LS scripts/ ==="
ls scripts/ | head -20
echo "=== ORIGINAL TRAINING SCRIPT ==="
find . -name "train.py" -o -name "train_*.py" -o -name "*train*3dgs*" 2>/dev/null | head -10
echo "=== DATA ==="
ls data/official/ | head -10
echo "=== DATA ROOM ==="
ls data/official/mipnerf360/ | head -10
