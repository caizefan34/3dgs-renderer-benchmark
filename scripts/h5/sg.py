import sys
import torch
sys.path = [p for p in sys.path if '/.local/' not in p]
paths = [p for p in sys.path if 'site-packages' in p]
print("site paths:")
for p in paths: print("  ", p)
try:
    import gsplat
    print("gsplat file:", gsplat.__file__)
    from gsplat.cuda import _backend
    print("backend _C:", type(_backend._C))
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAIL", repr(e)[:200])