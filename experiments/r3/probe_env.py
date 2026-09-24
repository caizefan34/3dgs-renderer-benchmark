#!/usr/bin/env python3
"""Probe local env for R3 measurement prerequisites."""
import os, sys, glob, json
import torch

print("=== ENV PROBE ===")
print(f"torch: {torch.__version__}")
print(f"cuda avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  cuda: {torch.version.cuda}")
    print(f"  mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    import gsplat
    print(f"gsplat: {getattr(gsplat, '__version__', 'unknown')}")
except Exception as e:
    print(f"gsplat import FAILED: {type(e).__name__}: {e}")

# Check for checkpoints
candidates = [
    "data/checkpoints",
    "baseline/reference_v1",
    "results/reference_v1/room_30k/checkpoints",
    "results/reference_v1",
    "results/",
]
print("\n=== CHECKPOINT SCAN ===")
for c in candidates:
    if os.path.isdir(c):
        pt_files = glob.glob(os.path.join(c, "**", "*.pt"), recursive=True)
        for f in pt_files[:10]:
            print(f"  {f}")
        if len(pt_files) > 10:
            print(f"  ... +{len(pt_files)-10} more")
    else:
        print(f"  (missing) {c}")

# Camera sequence
print("\n=== CAMERA SEQUENCE ===")
for c in ["data/camera_sequence.npy", "data/camera_presets/circle.json"]:
    print(f"  {c}: {'OK' if os.path.exists(c) else 'MISSING'}")

# Existing R3 outputs
print("\n=== R3 OUTPUTS ===")
if os.path.isdir("results/reference_v1/r3"):
    for f in sorted(os.listdir("results/reference_v1/r3")):
        print(f"  {f}")

# R3 module spot-check
print("\n=== R3 MODULES ===")
try:
    import r3_certificate_vaccine  # noqa
    print("  r3_certificate_vaccine: importable")
except ImportError:
    pass
