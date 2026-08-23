#!/usr/bin/env python3
"""
Check available 3DGS training infrastructure.

Inspects the environment for training code availability.
"""
import os
import sys
import shutil
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Python: {sys.executable}")
print()

# 1. Check if gsplat has training examples
try:
    import gsplat
    gsplat_path = os.path.dirname(gsplat.__file__)
    print(f"gsplat: {gsplat.__version__ if hasattr(gsplat, '__version__') else 'installed'}")
    print(f"gsplat path: {gsplat_path}")
    
    # Check for example scripts that include training
    examples_dirs = [
        os.path.join(gsplat_path, "..", "..", "..", "examples"),
        os.path.join(gsplat_path, "examples"),
    ]
    for ed in examples_dirs:
        ed = os.path.realpath(ed)
        if os.path.isdir(ed):
            print(f"  Examples: {ed}")
            for f in sorted(os.listdir(ed))[:20]:
                fpath = os.path.join(ed, f)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    print(f"    {f} ({size} bytes)")
except ImportError as e:
    print(f"gsplat not importable: {e}")
except Exception as e:
    print(f"gsplat error: {e}")

print()

# 2. Check for original 3DGS training code
search_paths = [
    os.path.join(REPO_ROOT, "..", "gaussian-splatting"),
    os.path.join(REPO_ROOT, "..", "3d-gaussian-splatting"),
    os.path.join(REPO_ROOT, "submodules", "3dgs"),
    os.path.join(REPO_ROOT, "external"),
    os.path.join(REPO_ROOT, "third_party"),
]
for sp in search_paths:
    sp = os.path.realpath(sp)
    if os.path.isdir(sp):
        files = sorted(os.listdir(sp))[:20]
        print(f"Found: {sp}")
        print(f"  Files: {files}")
        # Check for train.py
        train_py = os.path.join(sp, "train.py")
        if os.path.isfile(train_py):
            print(f"  HAS train.py!")

print()

# 3. Check for pip-installable 3DGS training library
print("Checking torch-ngp / simple-knn / diff-gaussian-rasterization...")
for pkg in ["diff_gaussian_rasterization", "simple_knn", "torch_ngp"]:
    spec = importlib.util.find_spec(pkg) if __name__ != "__main__" else None
    # Try direct import
    try:
        if pkg == "diff_gaussian_rasterization":
            import diff_gaussian_rasterization
            print(f"  {pkg}: available")
        elif pkg == "simple_knn":
            import simple_knn
            print(f"  {pkg}: available")
        else:
            exec(f"import {pkg}")
            print(f"  {pkg}: available")
    except ImportError:
        print(f"  {pkg}: NOT available")
    except Exception as e:
        print(f"  {pkg}: error - {e}")
