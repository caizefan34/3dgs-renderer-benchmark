#!/bin/bash
set -e
for env in /mnt/storage_pool/*/*-env /mnt/storage_pool/*-env /home/*/*-env /home/*-env; do
  [ -d "$env/bin" ] || continue
  py="$env/bin/python"
  [ -x "$py" ] || continue
  if "$py" -c "import gsplat" >/dev/null 2>&1; then
    echo "ENV=$env"
    echo "PY=$py"
    "$py" - <<'PYEOF'
import gsplat, torch
print("gsplat:", gsplat.__file__)
print("version:", getattr(gsplat, "__version__", "UNKNOWN"))
print("torch:", torch.__version__, torch.version.cuda)
print("experimental:", hasattr(gsplat, "experimental"))
from gsplat.experimental import rasterize_gaussian_higs_frozen
import inspect
print("frozen sig:", str(inspect.signature(rasterize_gaussian_higs_frozen))[:500])
PYEOF
    exit 0
  fi
done
echo "NO ENV FOUND"
exit 1
