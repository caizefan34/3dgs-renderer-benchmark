#!/usr/bin/env python3
"""P0: Verify binary identity"""
import sys
import os
import hashlib

print("=== P0: Binary Identity Verification ===\n")

# Check expected SHA256
EXPECTED_BASELINE = "40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50"
EXPECTED_CANDIDATE = "b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3"

# Find the actual binary
binary_paths = [
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/csrc.so",
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/cuda/csrc.so",
]

print("1. Searching for binary...")
found_binary = None
for path in binary_paths:
    if os.path.exists(path):
        print(f"   ✓ Found: {path}")
        found_binary = path
        break
    else:
        print(f"   ✗ Not found: {path}")

if found_binary:
    print(f"\n2. Computing SHA256...")
    with open(found_binary, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"   Actual SHA256: {sha256}")
    print(f"   Expected candidate: {EXPECTED_CANDIDATE}")
    print(f"   Expected baseline:  {EXPECTED_BASELINE}")
    
    if sha256 == EXPECTED_CANDIDATE:
        print("   ✓ MATCHES candidate")
    elif sha256 == EXPECTED_BASELINE:
        print("   ✓ MATCHES baseline")
    else:
        print("   ⚠ DOES NOT MATCH")
else:
    print("   ✗ No binary found")

# Check source file
print(f"\n3. Checking source file...")
source_file = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/cuda/csrc/IntersectTile.cu"
if os.path.exists(source_file):
    print(f"   ✓ Source exists: {source_file}")
    with open(source_file, 'rb') as f:
        source_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"   Source SHA256: {source_sha256}")
else:
    print(f"   ✗ Source not found")
