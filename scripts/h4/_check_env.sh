#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
echo "== python =="
$PY -c 'import sys; print(sys.version)'
echo "== numpy =="
$PY -c 'import numpy; print(numpy.__version__)'
echo "== torch =="
$PY -c 'import torch; print(torch.__version__, torch.version.cuda)'
echo "== gsplat =="
$PY -c 'import gsplat; print("gsplat ok")' 2>&1 | tail -2
echo "== plyfile =="
$PY -c 'import plyfile; print("plyfile ok")' 2>&1 | tail -2
echo "== GPU =="
$PY -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")' 2>&1 | tail -2
