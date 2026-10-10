#!/bin/bash
set +e
for ENV in nerficg strong_native anysplat; do
  PY=/home/liaoyuanjun/miniforge3/envs/$ENV/bin/python
  echo "=== env: $ENV ==="
  "$PY" - <<'PYEOF' 2>&1 | head -12
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("cxx11_abi", int(torch._C._GLIBCXX_USE_CXX11_ABI))
try:
    import gsplat; print("gsplat", getattr(gsplat,"__version__","?"))
except Exception as e:
    print("gsplat import failed:", repr(e)[:160])
try:
    import simple_knn._C; print("simple_knn OK")
except Exception as e:
    print("simple_knn:", repr(e)[:100])
PYEOF
done
echo "DONE"
