#!/usr/bin/env python3
"""Test subprocess finding cl.exe"""
import subprocess
import os

print(f"PATH: {os.environ.get('PATH', '')[:200]}", flush=True)
print(f"LIB: {os.environ.get('LIB', 'not set')}", flush=True)
print(f"INCLUDE: {os.environ.get('INCLUDE', 'not set')[:100]}", flush=True)

r = subprocess.run(["where", "cl"], capture_output=True)
print(f"where cl returncode: {r.returncode}", flush=True)
print(f"where cl stdout: {r.stdout.decode('utf-8', 'ignore')}", flush=True)
print(f"where cl stderr: {r.stderr.decode('utf-8', 'ignore')}", flush=True)
