set -eu
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
python3 - <<'PY'
from pathlib import Path
import zipfile
root=Path('/mnt/storage_pool/liaoyuanjun/wv0_raw_audit_20261008_2148').resolve()
with zipfile.ZipFile('/mnt/storage_pool/liaoyuanjun/wv0_audit_transfer_20261008.zip') as z:
 for name in z.namelist():
  target=(root/name).resolve()
  if not target.is_relative_to(root):raise RuntimeError('unsafe archive member')
 z.extractall(root)
PY
OUT=/mnt/storage_pool/liaoyuanjun/wv0_raw_audit_20261008_2148
python3 "$OUT/commands/build_audit.py" > "$OUT/logs/build_audit.stdout.log" 2> "$OUT/logs/build_audit.stderr.log"
nvidia-smi > "$OUT/gpu_snapshots/final_gpu.txt"
python3 "$OUT/commands/verify_and_package.py"
