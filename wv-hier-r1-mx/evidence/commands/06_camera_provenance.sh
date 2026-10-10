set -eu
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
python3 - <<'PY'
from pathlib import Path
import shutil,json,hashlib,subprocess
out=Path('/mnt/storage_pool/liaoyuanjun/wv0_raw_audit_20261008_2148')
repo=Path('/home/liaoyuanjun/3dgs-renderer-benchmark')
paths=['scripts/epic05/phase7/dataset.py','baseline/reference_v1/trainer.py','baseline/reference_v1/config.py','results/reference_v1/room_30k/provenance.json','results/reference_v1/s22/garden/provenance.json','results/reference_v1/s22/bicycle/provenance.json']
rows=[]
for name in paths:
 p=repo/name;row={'path':str(p),'exists':p.exists()}
 if p.exists():
  data=p.read_bytes();dst=out/'raw/camera_convention_source'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
  row.update({'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
 rows.append(row)
(out/'raw/camera_convention_files.json').write_text(json.dumps(rows,indent=2))
for sub in ['results/reference_v1/s22/garden','results/reference_v1/s22/bicycle']:
 p=repo/sub
 for f in p.glob('*.json'):
  if f.stat().st_size<200000:
   dst=out/'raw/camera_convention_source'/sub/f.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
print(json.dumps(rows,indent=2))
PY
