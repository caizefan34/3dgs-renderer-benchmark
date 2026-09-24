#!/usr/bin/env python3
"""Check R3 runner dependencies properly with utf-8 encoding."""
import sys, os, io
import torch

print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")

# Check the R3 runner script parses fine with utf-8
try:
    with io.open("experiments/r3/r3_certificate_runner.py", "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    compile(src, "r3_certificate_runner.py", "exec")
    print("r3_certificate_runner.py: parses OK")
except Exception as e:
    print(f"R3 runner parse issue: {type(e).__name__}: {e}")

# Check camera sequence listing
import glob
for p in glob.glob("data/**/*camera*", recursive=True) + glob.glob("data/camera_*"):
    print(f"  camera data: {p}")

# Check whether R3 runner can at least load its deps
sys.path.insert(0, "experiments/r3")
try:
    import r3_certificate_runner  # may fail on gsplat GPU deps
    print("r3_certificate_runner import: OK")
except Exception as e:
    print(f"r3_certificate_runner import: {type(e).__name__}: {e}")
