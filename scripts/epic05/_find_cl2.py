#!/usr/bin/env python3
"""Test if subprocess can find cl.exe with CUDA in PATH"""
import subprocess
import os

r = subprocess.run(["where", "cl"], capture_output=True)
print(f"where cl: returncode={r.returncode}, stdout={r.stdout.decode('utf-8','ignore').strip()}", flush=True)

# Also try with a full shell command
r2 = subprocess.run(["cmd", "/c", "where cl"], capture_output=True)
print(f"cmd /c where cl: returncode={r2.returncode}, stdout={r2.stdout.decode('utf-8','ignore').strip()}", flush=True)
