#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
echo "=== gsplat/torch ==="
$PY -c "import gsplat,torch; print('gsplat',gsplat.__version__,'torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'ngpu',torch.cuda.device_count())"
echo "=== official ply point counts ==="
for s in room garden bicycle; do
  $PY - "$s" <<'PYEOF'
import sys
from plyfile import PlyData
s=sys.argv[1]
p=PlyData.read(f'/tmp/3dgs-renderer-benchmark/data/official/mipnerf360/{s}/point_cloud.ply')
print(s,'points',len(p['vertex']))
PYEOF
done
echo "=== MX repo experiments/results listing ==="
ls /home/liaoyuanjun/3dgs-renderer-benchmark/experiments 2>/dev/null | head -20
echo "--- results ---"
ls /home/liaoyuanjun/3dgs-renderer-benchmark/results 2>/dev/null | head -20
