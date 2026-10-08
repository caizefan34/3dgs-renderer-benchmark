#!/bin/bash
set -e
cd /root/epic05_n1
cp -f N1_SPEC_OPERATIONALIZATION.json artifacts/N1_SPEC_OPERATIONALIZATION.json
/root/miniforge3/envs/gsplat/bin/python n1_manifest.py
echo "=== artifact listing ==="
ls -la artifacts/ | tail -32
