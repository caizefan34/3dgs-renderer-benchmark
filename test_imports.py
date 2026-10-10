#!/usr/bin/env python3
import os
import sys
import tempfile

# Test 1: tempfile
try:
    tmpdir = tempfile.gettempdir()
    print(f"TEMP_DIR: {tmpdir}")
    test_file = os.path.join(tmpdir, "test_tempfile.tmp")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print("TEMP_FILE_WRITE: PASS")
except Exception as e:
    print(f"TEMP_FILE_WRITE: FAIL - {e}")
    sys.exit(1)

# Test 2: PyTorch
try:
    import torch
    print(f"TORCH_VERSION: {torch.__version__}")
    print(f"CUDA_AVAILABLE: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA_VERSION: {torch.version.cuda}")
        print(f"GPU_NAME: {torch.cuda.get_device_name(0)}")
    print("TORCH_IMPORT: PASS")
except Exception as e:
    print(f"TORCH_IMPORT: FAIL - {e}")
    sys.exit(1)

# Test 3: gsplat
try:
    import gsplat
    print(f"GSPLAT_VERSION: {gsplat.__version__}")
    print(f"GSPLAT_PATH: {gsplat.__file__}")
    print("GSPLAT_IMPORT: PASS")
except Exception as e:
    print(f"GSPLAT_IMPORT: FAIL - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("ALL_IMPORTS: PASS")
