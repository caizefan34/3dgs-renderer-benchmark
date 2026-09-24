#!/usr/bin/env python3
"""Check camera data, R3 scripts, and GPU readiness."""
import os, glob, json

print("=== CAMERA / VIDEO DATA ===")
for pat in ["data/**/*.npy", "data/**/*.json", "data/**/*.jpg", "data/**/*.png"]:
    files = glob.glob(pat, recursive=True)
    for f in files[:15]:
        print(f"  {f}")
    if len(files) > 15:
        print(f"  ... +{len(files)-15} more")

print("\n=== R3 SCRIPTS ===")
for pat in ["experiments/r3/*.py", "experiments/r3/*.sh"]:
    files = glob.glob(pat)
    for f in files:
        sz = os.path.getsize(f)
        print(f"  {f} ({sz/1e3:.0f} KB)")

print("\n=== R3 MD/RPT ===")
for f in glob.glob("experiments/r3/*.md"):
    print(f"  {f}")
