#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
echo "== nvcc in higs env =="
find /mnt/storage_pool/liaoyuanjun/higs-13scene-env -name nvcc -type f 2>/dev/null | head
echo "== torch cuda home / cudart =="
$PY - <<'PYEOF'
import torch, os
print("cuda_home", torch.utils.cpp_extension.include for _ in [] if False else "n/a")
try:
    from torch.utils.cpp_extension import library_paths, include_paths
    print("libs", library_paths()[:3])
    print("incs", include_paths()[:3])
except Exception as e:
    print("ext path err", repr(e)[:150])
import glob
print("cudart glob:", glob.glob(os.path.join(os.path.dirname(torch.__file__),"lib","libcudart*")))
PYEOF
echo "== which nvcc / cuda =="
which nvcc; ls /usr/local/cuda*/bin/nvcc 2>/dev/null