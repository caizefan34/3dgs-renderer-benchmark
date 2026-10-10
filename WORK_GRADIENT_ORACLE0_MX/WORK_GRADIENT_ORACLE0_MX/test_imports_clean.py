#!/usr/bin/env python3
import os
import sys
import tempfile

print(f"TMPDIR: {os.environ.get('TMPDIR', 'NOT_SET')}")
print(f"tempfile.gettempdir(): {tempfile.gettempdir()}")

try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda version: {torch.version.cuda}")
        print(f"gpu: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import gsplat
    print(f"gsplat: {gsplat.__version__}")
    print(f"gsplat path: {gsplat.__file__}")
except Exception as e:
    print(f"gsplat import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("ALL IMPORTS OK")
