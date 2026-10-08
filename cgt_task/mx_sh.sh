#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import gsplat, inspect
# find SH eval + helpers
import gsplat.rendering as R
print("rendering funcs:", [n for n in dir(R) if not n.startswith('_')])
try:
    from gsplat import spherical_harmonics
    print("spherical_harmonics import OK")
except Exception as e:
    print("no top-level spherical_harmonics:", e)
# check fully_fused_projection return via docstring
print("--- projection doc ---")
print(gsplat.fully_fused_projection.__doc__[:600] if gsplat.fully_fused_projection.__doc__ else "none")
PYEOF
