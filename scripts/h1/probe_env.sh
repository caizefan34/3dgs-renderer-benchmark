#!/bin/bash
echo "=== ENV DIRS ==="
find /mnt/storage_pool /home -maxdepth 4 -type d -name "*env*" 2>/dev/null | head -30
echo "=== BENCHMARK FILES ==="
find /mnt/storage_pool /home -maxdepth 7 -name "run_*benchmark*.py" 2>/dev/null | head -10
echo "=== GSPlot location ==="
/mnt/storage_pool/liaoyunajun/himsg-13scene-env/bin/python -c "import gsslot, os; print(os.path.dirname(gsslot.__file__))" 2>&1 | head -3