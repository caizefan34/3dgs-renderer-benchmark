#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import torch, gsplat, numpy as np
# confirm gaussian_ids is the packed->original map
import inspect
print("rasterization returns meta with gaussian_ids (packed->orig) and flatten_ids (packed idx per intersection)")
PYEOF
