#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
echo "=== checkpoint structure (room 30k) ==="
$PY - <<'PYEOF'
import torch
ck=torch.load('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt', map_location='cpu', weights_only=False)
print('type', type(ck).__name__)
if isinstance(ck,dict):
    for k,v in ck.items():
        try: print(' ',k, tuple(v.shape) if hasattr(v,'shape') else (v if not hasattr(v,'__len__') else type(v).__name__))
        except Exception as e: print(' ',k,'ERR',e)
PYEOF
echo "=== gsplat per-tile API available ==="
$PY - <<'PYEOF'
import gsplat, inspect
print('gsplat', gsplat.__version__)
for name in ['rasterization','rasterization_to_indices','fully_fused_projection','isect_offset_encode','rasterize_to_pixels']:
    print(name, hasattr(gsplat,name))
PYEOF
echo "=== camera_sequence.npy shape ==="
$PY - <<'PYEOF'
import numpy as np
c=np.load('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/camera_sequence.npy')
print('camera_sequence', type(c).__name__, getattr(c,'shape',None), getattr(c,'dtype',None))
PYEOF
