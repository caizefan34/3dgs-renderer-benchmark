#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import gsplat, inspect
print("isect_tiles sig:")
print(inspect.signature(gsplat.isect_tiles))
print("--- doc ---")
print((gsplat.isect_tiles.__doc__ or "none")[:500])
PYEOF
