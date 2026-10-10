import importlib.util as u, subprocess, sys
# Is ninja importable as a python module (torch checks 'import ninja')?
for mod in ["ninja"]:
    s=u.find_spec(mod); print(mod, s.origin if s else "MISSING")
import os
# Try directly loading the prebuilt csrc.so
import torch
so="/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat/csrc.so"
try:
    spec=u.spec_from_file_location("gsplat_csrc_prebuilt", so)
    m=u.module_from_spec(spec)
    # need rich present
    try:
        spec.loader.exec_module(m)
        print("csrc.so LOADED OK; has _C:", hasattr(m,"_C"))
    except Exception as e:
        print("csrc.so load ERR", repr(e)[:300])
except Exception as e:
    print("spec ERR", repr(e)[:200])
