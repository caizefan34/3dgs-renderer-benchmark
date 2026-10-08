#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
R=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
echo "=== B1_room_30k config.json ==="
$PY -c "import json;d=json.load(open('$R/B1_room_30k/config.json'));print(json.dumps(d,indent=1)[:2000])"
echo "=== repo scene/colmap loaders ==="
grep -rl --include=*.py -e 'read_cameras_bin' -e 'def read_colmap' -e 'cameras.bin' -e 'images.bin' /home/liaoyuanjun/3dgs-renderer-benchmark 2>/dev/null | head -10
echo "=== nerfstudio/colmap libs available ==="
$PY -c "import importlib;[print(m, hasattr(importlib.import_module(m),'__version__')) for m in ['nerfview','pycolmap','nerfstudio']]" 2>&1 | head
