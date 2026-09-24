import ctypes, sys, os
# Try with RTLD_GLOBAL to make PyTorch symbols available
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
import torch
print("torch loaded")

# Now try loading the NEW .so directly
try:
    lib = ctypes.CDLL("/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/csrc.so", mode=ctypes.RTLD_NOW | ctypes.RTLD_GLOBAL)
    print("NEW .so loaded via ctypes: OK")
except Exception as e:
    print(f"NEW .so ctypes load FAILED: {e}")

# Try the ORIGINAL .so
try:
    lib2 = ctypes.CDLL("/tmp/gsplat_baseline/gsplat-1.5.3/build/lib.linux-x86_64-3.10/gsplat/csrc.so", mode=ctypes.RTLD_NOW | ctypes.RTLD_GLOBAL)
    print("ORIGINAL .so loaded via ctypes: OK")
except Exception as e:
    print(f"ORIGINAL .so ctypes load FAILED: {e}")

# Now try Python import with RTLD_GLOBAL
try:
    from gsplat.cuda._backend import _C
    print(f"_C loaded: {_C}")
except Exception as e:
    print(f"_C import FAILED: {e}")
