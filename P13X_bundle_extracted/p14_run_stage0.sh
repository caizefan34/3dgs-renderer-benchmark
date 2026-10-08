#!/bin/bash
mkdir -p /root/epic05_p14/logs
cd /root/epic05_p14/scripts
/root/miniforge3/envs/gsplat/bin/python p14_stage0_identity.py \
  > /root/epic05_p14/logs/stage0.log 2>&1
tail -3 /root/epic05_p14/logs/stage0.log
echo "--- identity verdict ---"
/root/miniforge3/envs/gsplat/bin/python -c "
import json
d = json.load(open('/root/epic05_p14/artifacts/P14_SOURCE_IDENTITY.json'))
print('verdict:', d['verdict_identity'])
for k, v in d['frozen_binaries'].items():
    print(f'  {k}: match={v[\"match\"]} sha={v.get(\"sha256\",\"?\")[:16]}...')
"
