#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
if command -v nvidia-smi >/dev/null 2>&1; then :; fi # do NOT call GPU in CPU-only kit
python3 test_properties.py | tee preflight.log
python3 cpu_oracle.py --size 40 --num 75 --seed 42 --output out/s42 | tee -a preflight.log
python3 cpu_oracle.py --size 40 --num 75 --seed 123 --output out/s123 | tee -a preflight.log
python3 - <<'PY'
from pathlib import Path
import hashlib,json
files=[]
for p in Path('.').rglob('*'):
    if p.is_file() and p.name!='MANIFEST_SHA256.json' and '__pycache__' not in p.parts:
        files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
Path('MANIFEST_SHA256.json').write_text(json.dumps({'files':sorted(files,key=lambda x:x['path'])},indent=2)+'\n')
PY
