import os, sys
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
C.bootstrap_c0()
import fused_ssim_cuda as F
print("DIR", [x for x in dir(F) if not x.startswith("__")])
import inspect
print("=== fusedssim doc ===")
print(getattr(F.fusedssim, "__doc__", None))
print("=== fusedssim_backward doc ===")
print(getattr(F.fusedssim_backward, "__doc__", None))
print("FUSEDSSIMMap file", inspect.getsourcefile(__import__("fused_ssim").FusedSSIMMap) if True else "")
