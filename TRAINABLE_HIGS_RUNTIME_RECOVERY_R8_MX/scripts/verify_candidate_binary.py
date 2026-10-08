#!/usr/bin/env python3
"""R8 P1: Verify candidate binary identity"""
import hashlib
import os

print("=== R8 P1: Binary Identity Verification ===\n")

# Expected SHA256
EXPECTED_CANDIDATE = "b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3"

# Check candidate binary
candidate_path = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/experimental/render/kernels/csrc.so"

print(f"1. Checking candidate binary...")
print(f"   Path: {candidate_path}")

if os.path.exists(candidate_path):
    print(f"   ✓ File exists")
    
    with open(candidate_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    
    print(f"   SHA256: {sha256}")
    print(f"   Expected: {EXPECTED_CANDIDATE}")
    
    if sha256 == EXPECTED_CANDIDATE:
        print(f"   ✓ MATCHES candidate")
    else:
        print(f"   ⚠ DOES NOT MATCH")
else:
    print(f"   ✗ File not found")
