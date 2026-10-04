#!/bin/bash
set -e
echo "--- manifest scenes ---"
grep -oE '"(room|bicycle|garden)"' /mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH/config/scene_manifest.json | sort -u
echo "--- probe syntax ---"
/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python - <<'PY'
import ast
ast.parse(open("/dev/shm/bes_opp/probe_opportunity.py").read())
print("syntax OK")
PY
echo "--- done ---"
