#!/bin/bash
set -x
cd /home/liaoyuanjun/3dgs-renderer-benchmark
python3 << 'PYEOF'
import inspect
from gsplat.cuda import _wrapper as w
src = inspect.getsource(w.spherical_harmonics)
print(src)
print("---FILE---")
print(w.__file__)
PYEOF